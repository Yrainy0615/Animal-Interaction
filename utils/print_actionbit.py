import sys

# --- 定数 ---
# 1つの行動に対応するプロンプトの行数
N_PROMPTS_PER_BEHAVIOR = 10
BEHAVIOR_FILE = '/mnt/nfs/mammal_net/annotation/behavior_to_id.txt'
ACTION_FILE = '/mnt/nfs/mammal_net/annotation/action_description_with_posebit.csv'

def parse_behaviors(behavior_file_content):
    """
    behavior_to_id.txt (の文字列内容) をパースし、
    IDをキー、行動名を値とする辞書を返す。
    """
    behaviors = {}
    lines = behavior_file_content.splitlines() # 改行で分割

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"[警告] スキップ: '{BEHAVIOR_FILE}' の行形式が不正です: {line}", file=sys.stderr)
            continue
            
        name, behavior_id_str = parts
        try:
            behavior_id = int(behavior_id_str)
            behaviors[behavior_id] = name
        except ValueError:
            print(f"[警告] スキップ: IDが数値ではありません: {line}", file=sys.stderr)
    
    return behaviors

def parse_actions(action_file_content):
    """
    action_description_with_posebit.csv (の文字列内容) をパースし、
    クリーンアップされたプロンプトのリストを返す。
    """
    lines = action_file_content.splitlines() # 改行で分割
    # 両端の空白と、もし " があれば除去 (例: "The Neck..." -> The Neck...)
    all_prompts = [line.strip().strip('"') for line in lines if line.strip()]
    return all_prompts

def group_and_print_behaviors(behaviors, all_prompts, n_prompts):
    """
    パースされたデータを受け取り、グループ化して出力する。
    """
    
    # 確実にID順に処理するため、IDでソートしたリストを作成
    sorted_behavior_ids = sorted(behaviors.keys())

    # --- データ整合性チェック ---
    expected_prompts = len(sorted_behavior_ids) * n_prompts
    if len(all_prompts) != expected_prompts:
        print(f"[警告] プロンプトの総数 ({len(all_prompts)}) が、"
              f"期待される数 ({len(sorted_behavior_ids)}行動 * {n_prompts}行 = {expected_prompts}行) "
              f"と一致しません。", file=sys.stderr)

    print("--- 行動ごとのプロンプトグループ ---")

    for i, behavior_id in enumerate(sorted_behavior_ids):
        behavior_name = behaviors[behavior_id]
        
        # N=n_prompts ごとにスライス
        start_index = i * n_prompts
        end_index = start_index + n_prompts
        
        # 該当するプロンプト群を取得
        prompts_group = all_prompts[start_index:end_index]
        
        # --- 出力 ---
        print(f"\n## {behavior_name} (ID: {behavior_id})")
        
        if not prompts_group:
            # 該当するプロンプトがなかった場合 (CSVデータが足りない)
            print("  [データなし]")
            continue

        for j, prompt in enumerate(prompts_group):
            print(f"  {j+1:02d}: {prompt}")

# --- メイン処理 (Try-Catch なし) ---

# 1. ファイル読み込み
# (ファイルが存在しない場合、ここで FileNotFoundError が発生して停止します)
with open(BEHAVIOR_FILE, 'r', encoding='utf-8') as f:
    behavior_data = f.read()
    
with open(ACTION_FILE, 'r', encoding='utf-8') as f:
    action_data = f.read()

# 2. behavior データをパース
behavior_map = parse_behaviors(behavior_data)

# 3. action データをパース
action_list = parse_actions(action_data)

# 4. グループ化して出力
if behavior_map and action_list:
    group_and_print_behaviors(behavior_map, action_list, N_PROMPTS_PER_BEHAVIOR)
else:
    print("[エラー] データのパースに失敗したか、ファイルが空です。", file=sys.stderr)