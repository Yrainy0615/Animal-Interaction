import csv
import json
import logging
import os
import re
from typing import List

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# --- 定数設定 ---

# 入力ファイルパス (行動リスト)
ACTION_LIST_PATH = "/mnt/nfs/lote/annotation/LoTE_label.csv"

# 出力ファイルパス (ファイル名を変更)
BASE_DIR = "/mnt/nfs/lote/annotation/"
ACTION_OUTPUT_CSV = os.path.join(BASE_DIR, "action_description_posebits_v2.csv")

# OpenAI API設定 (モデルとトークン数、Temperatureを変更)
# 警告: gpt-3.5-turbo-0125 では複雑な指示に従えない可能性が高いです。
# gpt-4o や gpt-4-turbo の使用を強く推奨します。
API_MODEL = "gpt-4o"  # "gpt-4-turbo" または "gpt-4o" を推奨
API_TEMPERATURE = 0.2  # 指示に厳格に従わせるため、値を下げる
API_MAX_TOKENS = 4096  # 10個のactionbit + JSON + レビューを生成するために増やす

# --- 行動タイプ定義 ---
BEHAVIOR_TYPES = {
    # === Interactive Behaviors (他個体との相互作用を含む行動) ===
    "Parental": "Interactive",    # 親子間の行動
    "Playing": "Interactive",     # 他個体との遊び（社会的遊び）と解釈
    "Grooming": "Interactive",    # 社会的グルーミング（他個体の手入れ）を含むと解釈
    "Aggregation": "Interactive", # 他個体と集まること
    "Mounting": "Interactive",    # 他個体への騎乗

    # === Solo Behaviors (基本的に単独で完結する行動) ===
    "Feeding": "Solo",            # 食事
    "Urinating": "Solo",          # 排尿
    "Resting": "Solo",            # 休息
    "Circumanal Gland Signing": "Solo", # マーキング（単独）
    "Urine Signing": "Solo",      # マーキング（単独）
    "Defecating": "Solo",         # 排便
    "Walking": "Solo",            # 歩行
    "Exploratory": "Solo",        # 探索
    "Drink Water": "Solo",        # 飲水
    "Smelling": "Solo",           # 匂い嗅ぎ（探索の一部）
    "Trotting": "Solo",           # 速歩
    "Climbing": "Solo",           # 登り
    "Jumping": "Solo",            # ジャンプ
    "Foraging": "Solo",           # 採食（探索的な食事）
    "Amusing": "Solo",            # 文脈不明だが、単独の遊び（例：おもちゃ）と推測

    # === 分類が困難な項目 ===
    "Miscellaneous": "Solo",      # 「その他」は特定のタイプに分類困難。
                                  # "Solo" (個体の状態) または "Unknown" が考えられるが、便宜上 "Solo" とした。
}

# 17個のキーポイントリスト (プロンプトに挿入するため)
KEYPOINT_LIST = (
    "Left Eye, Right Eye, Nose, Neck, Root of Tail, "
    "Left Shoulder, Left Elbow, Left Front Paw, "
    "Right Shoulder, Right Elbow, Right Front Paw, "
    "Left Hip, Left Knee, Left Back Paw, "
    "Right Hip, Right Knee, Right Back Paw"
)

# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_class_names(filepath: str) -> List[str]:
    """CSVファイルからクラス名（2列目）を読み込む (ヘッダーをスキップ)"""
    if not os.path.exists(filepath):
        logger.error("入力ファイルが見つかりません: %s", filepath)
        raise FileNotFoundError(f"入力ファイルが見つかりません: {filepath}")

    class_names: List[str] = []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            
            # ヘッダー行をスキップ
            try:
                header = next(reader)
                logger.info("CSVヘッダーをスキップしました: %s", header)
            except StopIteration:
                logger.warning("ファイルが空です: %s", filepath)
                return [] # 空のリストを返す

            # データ行を処理
            for row in reader:
                # [0, 'Parental'] のような行を想定
                if row and len(row) >= 2:
                    class_name = row[1].strip() # 2列目 (label) を取得
                    if class_name:
                        class_names.append(class_name)
                else:
                    logger.warning("不正な形式の行をスキップしました: %s", row)
                    
    except csv.Error as e:
        logger.error("CSVファイルの読み取り中にエラーが発生しました (%s): %s", filepath, e)
        raise
    except Exception as e:
        logger.error("ファイル読み取り中に予期せぬエラーが発生しました (%s): %s", filepath, e)
        raise

    logger.info("%s から %d 件のクラス名を読み込みました。", filepath, len(class_names))
    return class_names


def build_actionbit_prompt(class_name: str, behavior_type: str) -> str:
    """指定された行動ラベルとタイプに基づき、actionbit生成プロンプトを構築する"""

    if behavior_type == "Solo":
        policy_description = (
            "Policy 1 (Solo Behavior): Actionbit must describe the movement or state "
            "of a single individual's parts."
        )
        example_label = "Eat food"
        example_actionbits = (
            "The Neck lowers in the vertical direction., The Nose approaches the ground., "
            "Small vertical movements of the Neck are repeated., The Left Front Paw and Right Front Paw remain stationary., "
            "The Left Elbow and Right Elbow maintain a fixed high position., The Neck raises in the vertical direction., "
            "The Left Shoulder and Right Shoulder are stable., The Nose remains in close proximity to the ground for a duration., "
            "The Left Knee and Right Knee are flexed., The Root of Tail is stationary."
        )
    else:
        policy_description = (
            "Policy 2 (Interactive Behavior): Actionbit must describe the relative position, "
            "contact, or interaction between the parts of two individuals. It must use 'one individual' "
            "and 'the other individual'."
        )
        example_label = "Fight"
        example_actionbits = (
            "The Neck of one individual lowers., The Left Shoulder of one individual moves in the horizontal direction toward the other individual., "
            "The Right Shoulder of the other individual moves in the horizontal direction., High-velocity contact occurs between the Left Shoulder of one individual and the Right Shoulder of the other individual., "
            "The Nose of one individual is in close proximity to the Neck of the other individual., Both individuals' Left Hips and Right Hips show rapid horizontal movement., "
            "The Neck of one individual is positioned above the Neck of the other individual., The Left Front Paw of one individual raises vertically., "
            "The Right Front Paw of the other individual raises vertically., The Left Eye of one individual maintains proximity to the Right Shoulder of the other individual."
        )

    prompt = f"""
You are to perform a task: describe animal behaviors as physical transformations of keypoints (specific body parts). Your task is to convert ONE behavior label into a list of 10 distinct "actionbits."
An "actionbit" is defined as a short, descriptive text of a minimal, observable unit of movement or state of specific body parts that constitutes the behavior.

Your task is to generate descriptions ONLY for the behavior: "{class_name}"

You must generate descriptions for this 1 behavior according to the following rules.
Your response must consist of three distinct phases, generated in sequence. All output must be in English.

---
Phase 1: Actionbit List Generation (in English)
First, for the behavior "{class_name}", you must generate a list of 10 actionbits. This intermediate output must follow the exact format below, outputting the 10 actionbits as a single, comma-separated string.

Behavior Label: (The behavior label)
Actionbits: (10 actionbits, separated by commas. Each actionbit must adhere to the 'Actionbit Generation Policy' below.)

Actionbit Generation Policy (Applicable to 'Actionbits')
The behavior "{class_name}" is an "{behavior_type}" behavior.
{policy_description}

Vocabulary Restriction:
The vocabulary used to describe animal body parts must be limited ONLY to the following 17 types:
{KEYPOINT_LIST}
(Do not use terms not included in this 17-word list, e.g., head, body, Paws, Shoulders (Left/Right)).

Description Style and Viewpoint:
- Describe from an objective, third-person observer's viewpoint.
- Tautology is prohibited. Do not use the behavior name (e.g., "{class_name}") as a verb or noun within the actionbit.
- Describe the behavior as a physical transformation (i.e., "how the specified keypoints (parts) move").
- Select only the keypoints (from the 17-word list) that are important for the actionbit.
- Spatial Expression Rules: Use "Left" and "Right" exactly as they are included in the 17 keypoint names. Use objective coordinate expressions such as "horizontal direction" and "vertical direction."

Example of Phase 1 Output (Follow this format for your assigned behavior):
Behavior Label: {example_label}
Actionbits: {example_actionbits}

---
Phase 2: Final Output (English JSON Dictionary)
Second, immediately after completing Phase 1, you must output the final goal as a single JSON code block.
The JSON key must be the English behavior label "{class_name}" (string).
The JSON value must be an Array (list) of 10 strings, where each string is one actionbit generated in Phase 1.

Example of Phase 2 Output (Adhere to this format):
JSON
```json
{{
  "{example_label}": [
    "...",
    "..."
  ]
}}
```

-----

Phase 3: Self-Correction Review (Explicit Output)
Third, after outputting the JSON in Phase 2, you must output a review of that JSON as plain text. This review must explicitly state whether the Phase 2 output adheres to the rules.

  - Vocabulary Check: State whether all actionbit strings use body part names only from the specified 17-word list.
  - Count Check: State whether the behavior label in the JSON has an array (list) containing exactly 10 actionbit strings.
  - Interaction Terminology Check (Policy 2): For Interactive Behaviors, state whether the actionbit strings correctly use 'one individual' and 'the other individual'.
  - Tautology Check: State whether any actionbit uses the behavior name ("{class_name}") as a verb or noun.
    If all checks pass, state: Phase 3 Review: All rules adhered to.

-----

NOW, GENERATE ALL 3 PHASES FOR THE BEHAVIOR: "{class_name}"
"""
    return prompt.strip()


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def generate_actionbits_for_class(
    client: OpenAI,
    class_name: str,
    behavior_type: str,
    model: str,
    temp: float,
    max_len: int,
) -> List[str]:
    """OpenAI APIを呼び出して10個のactionbitリストを抽出する"""

    system_prompt = build_actionbit_prompt(class_name, behavior_type)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant specialized in generating precise "
                        "'actionbit' descriptions for computer vision tasks based on strict "
                        "vocabulary and format rules."
                    ),
                },
                {"role": "user", "content": system_prompt},
            ],
            temperature=temp,
            max_tokens=max_len,
            n=1,
        )

        full_response = completion.choices[0].message.content or ""

        json_match = re.search(r"```json\s*(\{.*?\})\s*```|(\{.*?\})", full_response, re.DOTALL)
        if not json_match:
            logger.error("レスポンスからJSONを抽出できませんでした (クラス: %s)。", class_name)
            raise ValueError("JSON output not found in API response")

        json_str = json_match.group(1) or json_match.group(2)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as error:
            logger.error("JSONのパースに失敗しました (クラス: %s): %s", class_name, error)
            raise ValueError(f"Failed to decode JSON: {error}") from error

        if not isinstance(data, dict) or not data:
            raise ValueError("JSON data is empty or not a dictionary")

        actionbits = data.get(class_name)
        if actionbits is None:
            logger.warning("JSONキー '%s' が見つかりません。最初のキーを使用します。", class_name)
            actionbits = next(iter(data.values()))

        if not isinstance(actionbits, list):
            raise ValueError("Parsed actionbits payload is not a list")

        if len(actionbits) != 10:
            logger.warning("抽出されたactionbitが10個ではありません (クラス: %s, 件数: %d)", class_name, len(actionbits))

        phase_3_match = re.search(r"Phase 3 Review:(.*)", full_response, re.DOTALL | re.IGNORECASE)
        if phase_3_match:
            review_text = phase_3_match.group(1).strip()
            logger.info("クラス '%s' のセルフレビュー: %s", class_name, review_text)
        else:
            logger.warning("Phase 3 レビューが見つかりません (クラス: %s)。", class_name)

        return actionbits

    except Exception as error:
        logger.warning("API呼び出し中にエラーが発生 (クラス: %s): %s", class_name, error)
        raise


def generate_actionbits_csv(
    client: OpenAI,
    input_path: str,
    output_path: str,
    model: str,
    temp: float,
    max_len: int,
) -> None:
    """入力ファイルに基づき、クラスごとのactionbitを生成してCSVに出力する"""

    logger.info("処理開始: %s -> %s", input_path, output_path)

    try:
        class_names = load_class_names(input_path)
    except FileNotFoundError:
        return

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Behavior_Label", "Caption_Number", "Caption"])

            with tqdm(total=len(class_names), desc=f"Generating {output_path}") as progress:
                for class_name in class_names:
                    behavior_type = BEHAVIOR_TYPES.get(class_name)
                    if not behavior_type:
                        logger.error("行動タイプ定義にクラス '%s' が見つかりません。", class_name)
                        writer.writerow([class_name, 0, "FAILED_UNKNOWN_BEHAVIOR_TYPE"])
                        progress.update(1)
                        continue

                    try:
                        actionbits = generate_actionbits_for_class(
                            client=client,
                            class_name=class_name,
                            behavior_type=behavior_type,
                            model=model,
                            temp=temp,
                            max_len=max_len,
                        )

                        for index, actionbit in enumerate(actionbits, start=1):
                            cleaned_actionbit = re.sub(r"\s+", " ", actionbit).strip().strip('"')
                            writer.writerow([class_name, index, cleaned_actionbit])

                    except Exception as error:
                        logger.error("クラス '%s' の生成に最終的に失敗しました: %s", class_name, error)
                        writer.writerow([class_name, 0, f"FAILED_GENERATION: {error}"])

                    progress.update(1)

        logger.info("正常に完了しました。出力ファイル: %s", output_path)

    except IOError as error:
        logger.error("出力ファイル '%s' への書き込みに失敗しました: %s", output_path, error)
    except Exception as error:
        logger.error("予期せぬエラーが発生しました: %s", error)


def main() -> None:
    """メインエントリポイント"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("環境変数 'OPENAI_API_KEY' が設定されていません。")
        return

    try:
        client = OpenAI(api_key=api_key)
    except Exception as error:
        logger.error("OpenAIクライアントの初期化に失敗しました: %s", error)
        return

    logger.warning(
        "これからOpenAI API (%s) を呼び出します。入力クラス数と同数のAPIコールが発生します。コストと時間に注意してください。",
        API_MODEL,
    )
    logger.warning(
        "必ず 'BEHAVIOR_TYPES' 辞書が '%s' の全クラス名をカバーしていることを確認してください。",
        ACTION_LIST_PATH,
    )

    generate_actionbits_csv(
        client=client,
        input_path=ACTION_LIST_PATH,
        output_path=ACTION_OUTPUT_CSV,
        model=API_MODEL,
        temp=API_TEMPERATURE,
        max_len=API_MAX_TOKENS,
    )


if __name__ == "__main__":
    main()