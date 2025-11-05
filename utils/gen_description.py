import os
import csv
import re
import time
from openai import OpenAI
from tqdm import tqdm
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

# --- 定数設定 ---

# 入力ファイルパス (ユーザー指定)
ANIMAL_LIST_PATH = "/mnt/nfs/mammal_net/annotation/genus_to_id.txt"
ACTION_LIST_PATH = "/mnt/nfs/mammal_net/annotation/behavior_to_id.txt"

# 出力ファイルパス
BASE_DIR = "/mnt/nfs/mammal_net/annotation/"
ANIMAL_OUTPUT_CSV = "animal_description.csv"
ACTION_OUTPUT_CSV = "action_description.csv"

ANIMAL_OUTPUT_CSV = os.path.join(BASE_DIR, ANIMAL_OUTPUT_CSV)
ACTION_OUTPUT_CSV = os.path.join(BASE_DIR, ACTION_OUTPUT_CSV)
# OpenAI API設定 (論文  に基づく)
API_MODEL = "gpt-3.5-turbo-0125"
API_TEMPERATURE = 0.99
# 論文  では "maximum token count to 77" と言及されているため
# max_tokens を 77 に設定する
API_MAX_TOKENS = 77 

# 論文  に基づくプロンプトテンプレート
# 動物 (Genus) 用
PROMPT_TEMPLATES_ANIMAL = [
    "Describe concisely what a {class_name} looks like;",
    "How can you identify a {class_name} concisely?;",
    "What does a {class_name} look like concisely?;",
    "What are the identifying characteristics of a {class_name}?;",
    "Please provide a concise description of the visual characteristics of {class_name}.",
]

# 行動 (Behavior) 用
PROMPT_TEMPLATES_ACTION = [
    "Describe concisely what animals doing {class_name} looks like;",
    "How can you identify animals doing {class_name} concisely?;",
    "What does animals doing {class_name} look like concisely?;",
    "What are the identifying characteristics of animals doing {class_name}?;",
    "Please provide a concise description of the visual characteristics of animals doing {class_name}.",
]

# 1クラスあたりの生成数 (5テンプレート * 10回 = 50)
GENERATIONS_PER_TEMPLATE = 10

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 関数定義 ---

def load_class_names(filepath: str) -> list[str]:
    """
    スペース区切りのリストファイルからクラス名（1列目）を読み込む
    """
    if not os.path.exists(filepath):
        logger.error(f"入力ファイルが見つかりません: {filepath}")
        raise FileNotFoundError(f"入力ファイルが見つかりません: {filepath}")
        
    class_names = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # 1つ以上の空白文字で分割し、最初の要素（クラス名）を取得
            parts = re.split(r'\s+', line_stripped)
            if parts:
                class_names.append(parts[0])
                
    logger.info(f"{filepath} から {len(class_names)} 件のクラス名を読み込みました。")
    return class_names

@retry(
    wait=wait_random_exponential(min=1, max=60), 
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((
        Exception  # OpenAI APIの特定のエラークラスを指定するのが望ましいが、ここでは一般的に捕捉
    )),
    reraise=True
)
def generate_single_description(
    client: OpenAI, 
    class_name: str, 
    template: str,
    model: str,
    temp: float,
    max_len: int
) -> str:
    """
    OpenAI APIを1回呼び出し、説明文を生成する（リトライ機能付き）
    """
    system_prompt = template.format(class_name=class_name)
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing concise visual descriptions for computer vision tasks."},
                {"role": "user", "content": system_prompt}
            ],
            temperature=temp,
            max_tokens=max_len,
            n=1
        )
        
        description = completion.choices[0].message.content
        
        # 改行や余分なスペースを削除して整形
        description_cleaned = re.sub(r'\s+', ' ', description).strip()
        return description_cleaned

    except Exception as e:
        logger.warning(f"API呼び出し中にエラーが発生 (クラス: {class_name}): {e}. リトライします...")
        raise

def generate_descriptions_for_file(
    client: OpenAI,
    input_path: str, 
    output_path: str, 
    templates: list[str], 
    model: str, 
    temp: float, 
    max_len: int
):
    """
    入力ファイルに基づき、クラスごとに50件の説明文を生成し、CSVに出力する
    """
    logger.info(f"処理開始: {input_path} -> {output_path}")
    
    try:
        class_names = load_class_names(input_path)
    except FileNotFoundError:
        return

    # CSVファイルへの書き込み
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ヘッダー行
            writer.writerow(["description"])
            
            # メインループ (クラスごとに処理)
            total_generations = len(class_names) * len(templates) * GENERATIONS_PER_TEMPLATE
            
            with tqdm(total=total_generations, desc=f"Generating {output_path}") as pbar:
                for class_name in class_names:
                    # テンプレートごとに処理
                    for template in templates:
                        # 1テンプレートあたり10回生成 [cite: 519]
                        for _ in range(GENERATIONS_PER_TEMPLATE):
                            try:
                                description = generate_single_description(
                                    client=client,
                                    class_name=class_name,
                                    template=template,
                                    model=model,
                                    temp=temp,
                                    max_len=max_len
                                )
                                # CSVに行を書き込む
                                writer.writerow([description])
                            except Exception as e:
                                logger.error(f"クラス '{class_name}' の生成に最終的に失敗しました: {e}")
                                # 失敗した場合でも、次の処理に進む（空行などを書き込んでも良い）
                                writer.writerow([f"FAILED_GENERATION_FOR_{class_name}"])
                            
                            pbar.update(1)
                            time.sleep(10)

        logger.info(f"正常に完了しました。出力ファイル: {output_path}")

    except IOError as e:
        logger.error(f"出力ファイル '{output_path}' への書き込みに失敗しました: {e}")
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}")

# --- メイン実行 ---

def main():
    # OpenAIクライアントの初期化 (環境変数 OPENAI_API_KEY を自動的に読み込む)
    try:
        client = OpenAI()
        # APIキーが設定されているか簡易チェック
        if client.api_key is None:
            logger.error("環境変数 'OPENAI_API_KEY' が設定されていません。")
            return
    except Exception as e:
        logger.error(f"OpenAIクライアントの初期化に失敗しました: {e}")
        return

    logger.warning(
        "これからOpenAI APIを呼び出します。"
        "入力クラス数に応じて大量のAPIコール（1クラスあたり50回）が発生するため、"
        "コストと時間に注意してください。"
    )

    # 1. 動物 (Genus) の説明文を生成
    generate_descriptions_for_file(
        client=client,
        input_path=ANIMAL_LIST_PATH,
        output_path=ANIMAL_OUTPUT_CSV,
        templates=PROMPT_TEMPLATES_ANIMAL,
        model=API_MODEL,
        temp=API_TEMPERATURE,
        max_len=API_MAX_TOKENS
    )

    # 2. 行動 (Behavior) の説明文を生成
    generate_descriptions_for_file(
        client=client,
        input_path=ACTION_LIST_PATH,
        output_path=ACTION_OUTPUT_CSV,
        templates=PROMPT_TEMPLATES_ACTION,
        model=API_MODEL,
        temp=API_TEMPERATURE,
        max_len=API_MAX_TOKENS
    )

if __name__ == "__main__":
    main()