import os
import torch
from yacs.config import CfgNode
import time 

# import sys
# sys.path.append('/home/nakagawa/compare_method/original/Animal-CLIP')
from models.model_builder import load_finetuned_xclip_model, xclip_load

# --- YACS Config ローダー (Animal-CLIP互換) ---
_C = CfgNode()
_C.DATA = CfgNode()
_C.DATA.ROOT = ""
_C.DATA.TRAIN_FILE = ""
_C.DATA.VAL_FILE = ""
_C.DATA.DATASET = ""
_C.DATA.NUM_FRAMES = 8
_C.DATA.NUM_CLASSES = 0
_C.DATA.LABEL_LIST = ""
_C.DATA.MULTI_CLASSES = False
_C.DATA.NUM_ANIMAL_CLASSES = 0
_C.DATA.ANIMAL_LABEL_LIST = ""
_C.DATA.RELATION_FILE = ""
_C.DATA.description = ""
_C.DATA.animal_description = ""

_C.MODEL = CfgNode()
_C.MODEL.ARCH = ""
_C.MODEL.MODEL_NAME = ""
_C.MODEL.HF_FINETUNED_PATH = ""
_C.MODEL.DROP_PATH_RATE = 0.0
_C.MODEL.FIX_TEXT = False
_C.MODEL.PROMPTS_ALPHA = 1e-1
_C.MODEL.PROMPTS_LAYERS = 3
_C.MODEL.MIT_LAYERS = 1

_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LOSS = ""
_C.TRAIN.ACCUMULATION_STEPS = 1
_C.TRAIN.EPOCHS = 30
_C.TRAIN.USE_CHECKPOINT = False

_C.PRED = False

def get_config(config_file: str) -> CfgNode:
    """
    YACS設定ノードをファイルからロードしてマージする。
    """
    config = _C.clone()
    if os.path.exists(config_file):
        config.merge_from_file(config_file)
    else:
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    config.freeze()
    return config

# --- 検証メイン関数 ---
def compare_models(cfg_base_path: str, cfg_ft_path: str):
    """
    xclip_load を2回実行してランダムキーを特定し、
    xclip_load と load_finetuned_xclip_model の確定的部分を比較検証する。
    """
    
    print("検証を開始します。")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    # --- Config ロード ---
    cfg_base = get_config(cfg_base_path)
    cfg_ft = get_config(cfg_ft_path)

    # --- 1. HFモデルのパス検証 ---
    hf_path = cfg_ft.MODEL.HF_FINETUNED_PATH
    print(f"\n--- ステップ1: HF事前学習済みモデルのパス検証 ---")
    print(f"HFモデルパス: {hf_path}")
    
    if not os.path.exists(hf_path):
        print(f"エラー: 指定されたHFモデルパスが存在しません: {hf_path}")
        print("Config (XCLIP-16-8-CONSTRUCT.yaml) の MODEL.HF_FINETUNED_PATH を確認してください。")
        print("検証を中止します。")
        return

    # --- 2. 比較のため、build_model に渡すパラメータを統一 ---
    use_cache_unified = not cfg_ft.MODEL.FIX_TEXT 
    
    params_for_build = {
        "T": cfg_ft.DATA.NUM_FRAMES,
        "droppath": cfg_ft.MODEL.DROP_PATH_RATE,
        "use_checkpoint": cfg_ft.TRAIN.USE_CHECKPOINT,
        "use_cache": use_cache_unified,
        "pred": cfg_ft.PRED,
        "prompts_alpha": cfg_ft.MODEL.PROMPTS_ALPHA,
        "prompts_layers": cfg_ft.MODEL.PROMPTS_LAYERS,
        "mit_layers": cfg_ft.MODEL.MIT_LAYERS,
        "logger": None,
    }
    
    print("\n--- 比較に使用する build_model 共通パラメータ ---")
    print(params_for_build)

    # --- 3. ランダムキー特定のため xclip_load を2回実行 ---
    
    # 3.1. xclip_load 1回目 (比較対象A)
    print(f"\n--- ステップ2: xclip_load (1回目) のロード ---")
    model_base_1, sd_base_1 = xclip_load(
        model_path=None, 
        name=cfg_base.MODEL.ARCH, 
        device=device,
        **params_for_build
    )
    print(f"xclip_load (1回目) 成功。 state_dict キー数: {len(sd_base_1)}")

    # 3.2. xclip_load 2回目 (ランダム検出用)
    print(f"\n--- ステップ3: xclip_load (2回目) のロード (ランダムキー検出用) ---")
    time.sleep(0.1) 
    model_base_2, sd_base_2 = xclip_load(
        model_path=None, 
        name=cfg_base.MODEL.ARCH, 
        device=device,
        **params_for_build
    )
    print(f"xclip_load (2回目) 成功。 state_dict キー数: {len(sd_base_2)}")

    # 3.3. ランダムキーの特定 (sd_base_1 vs sd_base_2)
    print(f"\n--- ステップ4: ランダムキーの特定 ---")
    random_keys = set()
    
    common_keys_base = set(sd_base_1.keys()) & set(sd_base_2.keys())
    if len(common_keys_base) != len(sd_base_1):
        print(f"警告: 1回目と2回目でキーセットが異なります。")

    for key in sorted(list(common_keys_base)):
        tensor_1 = sd_base_1[key].cpu()
        tensor_2 = sd_base_2[key].cpu()
        
        if tensor_1.dtype != tensor_2.dtype:
            random_keys.add(key)
            continue
            
        if not torch.allclose(tensor_1, tensor_2, atol=1e-6):
            random_keys.add(key)
    
    print(f"ランダム初期化キー (と推測されるキー) を {len(random_keys)} 個特定しました。")
    if len(random_keys) > 0 and len(random_keys) < 20: 
        print(f"  (例: {sorted(list(random_keys))[0]})")
    
    del model_base_2, sd_base_2

    # --- 4. モデルB (load_finetuned_xclip_model) のロード ---
    print(f"\n--- ステップ5: load_finetuned_xclip_model (比較対象B) のロード ---")

    model_ft = load_finetuned_xclip_model(
        hf_model_path=hf_path,
        device=device,
        xclip_params=params_for_build,
        expected_arch=cfg_ft.MODEL.ARCH
    )
    sd_ft = model_ft.state_dict()
    print(f"load_finetuned_xclip_model 成功。 state_dict キー数: {len(sd_ft)}")
    
    del model_base_1, model_ft

    # --- 5. state_dict の比較 (sd_base_1 vs sd_ft) ---
    print(f"\n--- ステップ6: state_dict の比較検証 (ランダムキー除外) ---")

    keys_base_1 = set(sd_base_1.keys())
    keys_ft = set(sd_ft.keys())

    diff_keys_base_only = keys_base_1 - keys_ft
    # --- ここを修正 ---
    diff_keys_ft_only = keys_ft - keys_base_1
    # -------------------
    common_keys_final = keys_base_1 & keys_ft

    # 5.1. キーの比較
    is_keys_identical = True
    print(f"\n[キーの比較]")
    if not diff_keys_base_only and not diff_keys_ft_only:
        print(f"OK: キーセットは同一です (共通キー {len(common_keys_final)} 個)。")
    else:
        is_keys_identical = False
        print("NG: キーセットが異なります。")
        if diff_keys_base_only:
            print(f"  xclip_load (Base) のみに存在するキー: {diff_keys_base_only}")
        if diff_keys_ft_only:
            print(f"  load_finetuned (FT) のみに存在するキー: {diff_keys_ft_only}")

    # 5.2. テンソル値の比較 (確定的キーのみ)
    print(f"\n[テンソル値の比較 (確定的キー)]")
    mismatched_tensors = []
    
    keys_to_compare = common_keys_final - random_keys
    
    print(f"  (共通キー {len(common_keys_final)} 個のうち、")
    print(f"   ランダムキー {len(random_keys)} 個を除外した")
    print(f"   確定的キー {len(keys_to_compare)} 個を比較)")

    is_deterministic_tensors_identical = True 

    for key in sorted(list(keys_to_compare)):
        tensor_base = sd_base_1[key].cpu()
        tensor_ft = sd_ft[key].cpu()
        
        if tensor_base.dtype != tensor_ft.dtype:
            is_deterministic_tensors_identical = False
            mismatched_tensors.append(
                f"  - {key} (型不一致: Base {tensor_base.dtype} vs FT {tensor_ft.dtype})"
            )
            continue
            
        if not torch.allclose(tensor_base, tensor_ft, atol=1e-6):
            is_deterministic_tensors_identical = False
            diff = torch.dist(tensor_base.float(), tensor_ft.float())
            abs_diff = torch.max(torch.abs(tensor_base.float() - tensor_ft.float()))
            mismatched_tensors.append(
                f"  - {key} (値不一致: L2-Dist = {diff.item():.8f}, Max-Abs-Diff = {abs_diff.item():.8f})"
            )

    if not mismatched_tensors:
        print(f"OK: 確定的キー {len(keys_to_compare)} 個すべてのテンソル値が一致しました (atol=1e-6)。")
    else:
        print(f"NG: 確定的キーで {len(mismatched_tensors)} 個のテンソルで値または型が不一致です。")
        for mismatch in mismatched_tensors[:20]:
            print(mismatch)
        if len(mismatched_tensors) > 20:
            print(f"  ... (他 {len(mismatched_tensors) - 20} 件の不一致)")

    # --- 6. 最終結果 ---
    print("\n" + "="*30)
    print("      検証結果")
    print("="*30)

    if is_keys_identical and is_deterministic_tensors_identical:
        print("結論: 合致 (Identical)")
        print("ランダム初期化部分を除き、確定的部分（標準CLIP重みなど）は")
        print("両方のロード方法で同一であることが確認されました。")
    else:
        print("結論: 不合致 (Mismatch)")
        if not is_keys_identical:
             print("原因: state_dict のキーセットが異なります。")
        if not is_deterministic_tensors_identical:
             print("原因: 確定的部分（標準CLIP重みなど）のテンソル値が異なります。")
             print("       HF -> OpenAI の変換ロジックに誤りがある可能性があります。")

    if len(random_keys) > 0:
        print("\n---")
        print("注記: 以下のキーはランダム初期化と判断し、比較から除外しました。")
        print(f"  (計 {len(random_keys)} 個)")
        representative_keys = sorted(list(set(
            k.split('.')[0] if '.' in k else k for k in random_keys
        )))
        print(f"  (影響範囲: {representative_keys} モジュールなど)")


if __name__ == "__main__":
    CFG_BASE_PATH = "/home/nakagawa/compare_method/original/Animal-CLIP/configs/mmnet/XCLIP-16-8.yaml"
    CFG_FT_PATH = "/home/nakagawa/compare_method/original/Animal-CLIP/configs/mmnet/XCLIP-16-8-CONSTRUCT.yaml"

    compare_models(CFG_BASE_PATH, CFG_FT_PATH)