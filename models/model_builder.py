import os
from typing import Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from .mit import MultiframeIntegrationTransformer, ResidualAttentionBlock
from .prompt import AnimalSpecificPrompt
from .cct import CrossFrameCommunicationTransformer
import sys
import warnings
sys.path.append("../")
from clip.model import CLIP,LayerNorm,Transformer,VisionTransformer
import clip
from transformers import CLIPModel
import math
from functools import partial
from torch.nn.init import trunc_normal_
import torch.nn.functional as F

import utils.weight_init_helper as init_helper
from .attention import MultiScaleBlock
from .batchnorm_helper import get_norm
from .stem_helper import PatchEmbed
from .utils import round_width
from .tools import plot_attention

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

from functools import reduce
import operator

import einops

from .attention import TemporalCrossAttention

from clip.model import QuickGELU, Attention, CLIP, LayerNorm, Transformer
from .weight_loaders import weight_loader_fn_dict
from .vision_transformer import (
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)

from typing import Dict, Iterable, List, Tuple, Union, Optional

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import TimesformerModel, CLIPTokenizer, CLIPTextModel, CLIPVisionModel, logging


class TemporalTransformer(nn.Module):
    """
    (B, T, E) のフレーム特徴量シーケンスを受け取り、
    各フレームを時系列で変換した「精錬されたフレーム列 (B, T, E)」を返す。
    以前の[CLS]による(B, E)集約はやめ、時間方向アテンション後の各トークンを返す。
    """
    def __init__(self, T: int, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers

        # XCLIPのResidualAttentionBlockを使用 (attn_mask=None)
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask=None) for _ in range(layers)]
        )

        # [CLS] トークン（互換のため保持、返却はしない）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))

        # 位置エンコーディング (Tフレーム + [CLS]トークン)
        self.positional_embedding = nn.Parameter(torch.empty(1, T + 1, width))

        self.ln_final = LayerNorm(width)

        # パラメータの初期化
        nn.init.normal_(self.cls_token, std=0.01)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def forward(self, x: torch.Tensor):
        # x: (B, T, E)
        b = x.shape[0]

        # [CLS] トークンを追加 (B, 1, E)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, E)

        # 位置エンコーディングを追加
        x = x + self.positional_embedding  # (B, T+1, E)

        # (B, T+1, E) -> (T+1, B, E) (ResidualAttentionBlockの入力形式)
        x = x.permute(1, 0, 2)

        # Transformer
        x = self.resblocks(x)

        # (T+1, B, E) -> (B, T+1, E)
        x = x.permute(1, 0, 2)

        # 全トークンにLayerNormを適用
        x = self.ln_final(x)

        # [CLS]を除いたフレームトークンのみ返す (B, T, E)
        x = x[:, 1:, :]

        return x

class SemanticFuser(nn.Module):
    """
    Cross-Attention機構 (Transformer Decoder Layer)。
    Q (行動クエリ) が K/V (動物コンテキスト) を参照し、
    セマンティックが融合されたベクトルを生成する。
    """
    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        
        # Q (行動), K (動物), V (動物)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True
        )
        
        # Decoder Layer に倣い、FeedForward と LayerNorm を追加
        self.ln_1 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln_2 = LayerNorm(embed_dim)
        
    def forward(self, 
                query_vectors: torch.Tensor, # (B, N, E) - 行動クエリ (Q)
                context_vectors: torch.Tensor # (B, 1, E) - 動物コンテキスト (K, V)
               ):
        
        # 1. Cross-Attention (QがK/Vを参照)
        # Q=(B, N, E), K=(B, 1, E), V=(B, 1, E)
        attn_output, _ = self.cross_attn(
            query=query_vectors, 
            key=context_vectors, 
            value=context_vectors
        )
        
        # 2. Add & Norm (Residual)
        x = self.ln_1(query_vectors + attn_output)
        
        # 3. FeedForward & Add & Norm (Residual)
        x = self.ln_2(x + self.mlp(x))
        
        return x # (B, N, E) - 融合された行動ベクトル

class ActionFuser(nn.Module):
    """
    Cross-Attention機構 (Transformer Decoder Layer)。
    Q (Description) が K/V (Actionラベル) を参照し、
    セマンティックが融合された行動ベクトルを生成する。
    
    入力: Q(N, E), K/V(M, E) -> 出力: (N, E)
    """
    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        
        # batch_first=True を使用
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, nhead, dropout=dropout, batch_first=True
        )
        
        self.ln_1 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ln_2 = LayerNorm(embed_dim)
        
    def forward(self, 
                query_vectors: torch.Tensor,   # (N, E) - Description (Q)
                context_vectors: torch.Tensor  # (M, E) - Action (K, V)
               ):
        
        # nn.MultiheadAttention (batch_first=True) のため
        # (N, E) -> (1, N, E)
        # (M, E) -> (1, M, E)
        q = query_vectors.unsqueeze(0)
        k = context_vectors.unsqueeze(0)
        v = context_vectors.unsqueeze(0)
        
        # 1. Cross-Attention
        # attn_output: (1, N, E)
        attn_output, _ = self.cross_attn(query=q, key=k, value=v)
        
        # 2. Add & Norm (Residual)
        x = self.ln_1(q + attn_output)
        
        # 3. FeedForward & Add & Norm (Residual)
        x = self.ln_2(x + self.mlp(x))
        
        # (1, N, E) -> (N, E)
        return x.squeeze(0)

class FrameFeatureAdapter(nn.Module):
    """
    CLIP-Adapter（local_contrast_adapter.py で学習した Adapter）を
    フレーム特徴に適用する単一のアダプタ。
    - 重みがロードできた場合のみ「アクティブ」になり、(1-a):a でブレンド
    - 重みが無い/ロード失敗時は「非アクティブ」= 完全に通さない・学習もしない
    保存側の構造に合わせて、以下の2層ボトルネックMLPを内部に持つ:
      fc: Linear(c_in, hidden, bias=False) -> ReLU -> Linear(hidden, c_in, bias=False) -> ReLU
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.fc: Optional[nn.Sequential] = None  # 後でロード時に形状に合わせて組み立て
        self.register_buffer("alpha", torch.tensor(0.5, dtype=torch.float32), persistent=True)
        self._active: bool = False

    @property
    def active(self) -> bool:
        return self._active and (self.fc is not None)

    def _build(self, c_in: int, hidden: int):
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def load_from_local_contrast(
        self,
        directory: str,
        state_filename: str = "global_adapter_state.pt",
        trainable: bool = False,
    ) -> bool:
        """
        local_contrast_adapter.py の save_model() が保存した state を読み込み:
          - "adapter" の state_dict をロード
          - config.ADAPTER_RATIO があれば alpha を上書き
        成功時 True を返し、以後 active になる。失敗時は非アクティブのまま。
        """
        try:
            state_path = os.path.join(directory, state_filename)
            if not os.path.isfile(state_path):
                print(f"[FrameFeatureAdapter] State file not found: {state_path}")
                self._active = False
                return False

            state = torch.load(state_path, map_location="cpu")
            adapter_sd = state.get("adapter", None)
            if adapter_sd is None:
                print(f"[FrameFeatureAdapter] 'adapter' state not found in {state_path}")
                self._active = False
                return False

            # 形状を推定してfcを構築
            # fc.0.weight: (hidden, c_in), fc.2.weight: (c_in, hidden)
            w0 = adapter_sd.get("fc.0.weight", None)
            w2 = adapter_sd.get("fc.2.weight", None)
            if w0 is None or w2 is None:
                # --- キーが 'fc.' 始まりでない場合のフォールバック（修正案） ---
                # もし '0.weight' が直接存在すれば、それを使う
                w0_alt = adapter_sd.get("0.weight", None)
                w2_alt = adapter_sd.get("2.weight", None)
                
                if w0_alt is not None and w2_alt is not None:
                    print("[FrameFeatureAdapter] Note: Found adapter keys without 'fc.' prefix.")
                    w0, w2 = w0_alt, w2_alt
                else:
                    print(f"[FrameFeatureAdapter] Unexpected state_dict keys in {state_path}. Expected 'fc.0.weight' or '0.weight'.")
                    self._active = False
                    return False
                # --- フォールバックここまで ---

            hidden = int(w0.shape[0])
            c_in = int(w0.shape[1])
            if c_in != self.embed_dim:
                # フレーム特徴次元と一致しない場合は使えない
                print(f"[FrameFeatureAdapter] Dim mismatch: adapter c_in={c_in} vs embed_dim={self.embed_dim}")
                self._active = False
                return False

            self._build(c_in=c_in, hidden=hidden)
            
            cleaned_sd = {
                k.replace("fc.", ""): v 
                for k, v in adapter_sd.items() 
                if k.startswith("fc.")
            }
            
            # 変換後の state_dict をロードする
            # もし cleaned_sd が空（fc.プレフィックスが無かった）なら、元の adapter_sd を試す
            load_sd = cleaned_sd if cleaned_sd else adapter_sd
            self.fc.load_state_dict(load_sd, strict=True)

            # ratio
            cfg = state.get("config", {})
            ar = cfg.get("ADAPTER_RATIO", None)
            if ar is not None:
                with torch.no_grad():
                    self.alpha.copy_(torch.tensor(float(ar), dtype=torch.float32))

            # 学習可否
            for p in self.parameters():
                p.requires_grad = bool(trainable)

            print(f"[FrameFeatureAdapter] Loaded adapter from {state_path} (alpha={float(self.alpha):.4f}, trainable={trainable})")
            self._active = True
            return True
        except Exception as e:
            # load_state_dict のエラーメッセージ (Missing keys, Unexpected keys) もここでキャッチされる
            print(f"[FrameFeatureAdapter] Failed to load adapter: {e}")
            self._active = False
            return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return x
        adapted = self.fc(x)
        a = float(self.alpha.item())
        return (1.0 - a) * x + a * adapted

class VideoChangeWeightedCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 T: int = 8,
                 temporal_layers: int = 2,
                 temporal_heads: int = 8,
                 device: str = "cuda",
                 use_logit_scale: bool = True,
                 external_adapter_dir: str = "",
                 external_state_filename: str = "global_adapter_state.pt",
                 external_adapter_trainable: bool = False):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        if temporal_heads <= 0:
            temporal_heads = transformer_heads

        self.temporal_transformer = TemporalTransformer(
            T=T,
            width=embed_dim,
            layers=temporal_layers,
            heads=temporal_heads
        )

        self.semantic_fuser = SemanticFuser(embed_dim=embed_dim, nhead=temporal_heads)
        self.action_fuser = ActionFuser(embed_dim=embed_dim, nhead=temporal_heads)

        # 理論 (数式 15-19) に必要なレイヤーの定義
        self.dk_dim = embed_dim 
        self.Wq_v = nn.Linear(embed_dim, self.dk_dim, bias=False) 
        self.Wk_v = nn.Linear(embed_dim, self.dk_dim, bias=False) 
        self.Wv_v = nn.Linear(embed_dim, embed_dim, bias=False)   
        self.Ws = nn.Linear(embed_dim, embed_dim, bias=False) 
        self.norm_1 = nn.LayerNorm(embed_dim) 
        self.norm_2 = nn.LayerNorm(embed_dim) 
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), 
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.frame_adapter = FrameFeatureAdapter(embed_dim=embed_dim)
        if external_adapter_dir:
            loaded = self.frame_adapter.load_from_local_contrast(
                directory=external_adapter_dir,
                state_filename=external_state_filename,
                trainable=external_adapter_trainable
            )
            if not loaded:
                pass # ロード失敗時はそのまま非アクティブ（スキップ）

        self.register_buffer("animal_vectors_lookup_table", None, persistent=False)
        self.use_logit_scale = bool(use_logit_scale)
        self.freeze_clip_encoders()

    def freeze_clip_encoders(self):
        for param in self.visual.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.token_embedding.parameters():
            param.requires_grad = False
        self.positional_embedding.requires_grad = False
        for param in self.ln_final.parameters():
            param.requires_grad = False
        self.text_projection.requires_grad = False

    def _encode_frames_raw(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        動画フレーム (B, T, C, H, W) -> CLIP 画像エンコーダ -> (B, T, D)
        """
        b, t, c, h, w = video_frames.shape
        dtype_img = torch.float32 
        with torch.no_grad():
            video_frames_flat = video_frames.type(dtype_img).reshape(-1, c, h, w)
            frame_features = self.encode_image(video_frames_flat)  # (B*T, D)
            frame_features = frame_features.reshape(b, t, -1)      # (B, T, D)
        return frame_features

    def encode_video_frames(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        - Original CLIP: CLIP -> features
        - CLIP-Adapter: CLIP -> Adapter blend -> features
        """
        frame_features = self._encode_frames_raw(video_frames).float()
        if self.frame_adapter.active:
            frame_features = self.frame_adapter(frame_features)
        return frame_features  # (B, T, D)

    @torch.no_grad()
    def precompute_animal_lookup(self, animal_labels: torch.Tensor, device="cuda", batch_size: int = 64):
        print("--- Pre-computing Animal Lookup Table (with mini-batching) ---")
        text_encoder_parts = [
            self.token_embedding, self.positional_embedding,
            self.transformer, self.ln_final, self.text_projection
        ]
        original_devices = {}
        try:
            for part in text_encoder_parts:
                if isinstance(part, nn.Module):
                    original_devices[part] = next(part.parameters(), torch.tensor(0)).device
                else:
                    original_devices[part] = part.device
                part.to(device)
        except StopIteration:
            pass

        if animal_labels.dim() == 3:
            a_classes, a_prompts, _ = animal_labels.shape
            animal_input = animal_labels.view(a_classes * a_prompts, -1)
        else:
            a_classes = animal_labels.shape[0]
            a_prompts = 1
            animal_input = animal_labels

        total_inputs = animal_input.shape[0]
        all_features = []
        print(f"Encoding {total_inputs} animal prompts in batches of {batch_size}...")
        for i in range(0, total_inputs, batch_size):
            batch = animal_input[i: i + batch_size].to(device)
            batch_features = self.encode_text(batch)
            all_features.append(batch_features.cpu())

        lookup_table = torch.cat(all_features, dim=0)
        print("Encoding complete.")

        if a_prompts > 1:
            lookup_table = lookup_table.view(a_classes, a_prompts, -1)
            lookup_table, _ = torch.max(lookup_table, dim=1)

        for part in text_encoder_parts:
            if part in original_devices:
                part.to(original_devices[part])

        self.animal_vectors_lookup_table = lookup_table.float()
        print(f"--- Animal Lookup Table pre-computed. Shape: {tuple(self.animal_vectors_lookup_table.shape)} ---")

    def forward(self,
                video_frames: torch.Tensor,
                text: torch.Tensor,
                animal_labels: torch.Tensor,
                description_labels: torch.Tensor,
                *args, **kwargs):
        """
        類似度計算（理論: Part-category-specific visual promptingに基づく）
        """
        if not args:
            raise ValueError("Missing required argument 'animal_pred' in *args for Lookup Mode.")
        animal_pred = args[0]
        B = video_frames.shape[0]

        # 1) フレーム特徴（Adapter適用は encode_video_frames 内）
        frame_features = self.encode_video_frames(video_frames)  # (B, T, D)
        T_len = frame_features.shape[1]
        D = frame_features.shape[-1] # D次元

        # 2) Temporal Transformer
        tt_out = self.temporal_transformer(frame_features.float())
        if tt_out.dim() == 3 and tt_out.shape[:2] == (B, T_len):
            refined_frames = tt_out  # V: (B, T, D)
            video_vector = refined_frames.mean(dim=1)  # 返却用（互換）
        else:
            refined_frames = frame_features  # フォールバック
            video_vector = tt_out if tt_out.dim() == 2 else frame_features.mean(dim=1)

        # 3) テキスト側（行動/説明）+ 動物コンテキスト
        with torch.no_grad():
            # 行動テキスト（Q）
            if text.dim() == 3:
                n_classes, n_prompts, _ = text.shape
                text_input = text.view(n_classes * n_prompts, -1)
            else:
                n_classes, _ = text.shape
                n_prompts = 1
                text_input = text
            action_vectors_q = self.encode_text(text_input)
            if n_prompts > 1:
                action_vectors_q = action_vectors_q.view(n_classes, n_prompts, -1)
                action_vectors_q, _ = torch.max(action_vectors_q, dim=1)
            N = n_classes

            # 説明テキスト（K/V）
            if description_labels.dim() == 3:
                n_desc, n_prompts_desc, _ = description_labels.shape
                desc_input = description_labels.view(n_desc * n_prompts_desc, -1)
            else:
                n_desc, _ = description_labels.shape
                n_prompts_desc = 1
                desc_input = description_labels
            description_vectors_kv = self.encode_text(desc_input)
            if n_prompts_desc > 1:
                description_vectors_kv = description_vectors_kv.view(n_desc, n_prompts_desc, -1)
                description_vectors_kv, _ = torch.max(description_vectors_kv, dim=1)

            if N != n_desc:
                raise ValueError(
                    f"Action(Q)/Description(K/V) class count mismatch: text(Q) ({N}) vs description_labels(K/V) ({n_desc})."
                )

            # 動物コンテキスト（事前計算テーブル）
            if self.animal_vectors_lookup_table is None:
                raise ValueError("Animal vectors lookup table is not pre-computed. Call precompute_animal_lookup() first.")
            
            lookup_table_cpu = self.animal_vectors_lookup_table
            a_classes = lookup_table_cpu.shape[0]
            if B != animal_pred.shape[0]:
                raise ValueError(f"Batch size mismatch: video_frames ({B}) vs animal_pred ({animal_pred.shape[0]})")
            if a_classes != animal_pred.shape[1]:
                raise ValueError(f"Animal class mismatch: lookup ({a_classes}) vs animal_pred ({animal_pred.shape[1]})")
            
            current_device = animal_pred.device
            lookup_table = lookup_table_cpu.to(current_device)

            animal_pred_expanded = animal_pred.unsqueeze(1) 
            lookup_table_expanded = lookup_table.unsqueeze(0).expand(B, -1, -1) 
            animal_vectors = torch.bmm(
                animal_pred_expanded.to(lookup_table_expanded.dtype),
                lookup_table_expanded
            )  # (B, 1, D)

        # 4) Fuser で「最終テキスト特徴」を作る（学習対象）
        action_query_vectors = self.action_fuser(
            query_vectors=action_vectors_q.float(),
            context_vectors=description_vectors_kv.float()
        ) 

        Q = action_query_vectors.unsqueeze(0).expand(B, -1, -1) 
        K = animal_vectors                                      
        fused_action_vectors = self.semantic_fuser(query_vectors=Q, context_vectors=K) # A': (B, N, D)
        
        # --- 5) Part-category-specific visual prompting (数式 15-19) ---
        
        refined_frames = refined_frames.float()
        fused_action_vectors = fused_action_vectors.float()
        
        # 5-1. 射影 (数式 15): Q_v, K_v, V_v の計算
        Q_v = self.Wq_v(fused_action_vectors) # Q_v: (B, N, d_k)
        K_v = self.Wk_v(refined_frames)       # K_v: (B, T, d_k)
        V_v = self.Wv_v(refined_frames)       # V_v: (B, T, D)
        
        # 5-2. Scaled Dot-Product Attention (数式 16): 重み $\alpha$ の計算
        # 類似度計算 Q_v K_v^T: (B, N, T)
        sim_bnt = torch.bmm(Q_v, K_v.transpose(1, 2))
        
        # スケール項の適用 (数式 16 の分母 $\sqrt{d_k}$)
        dk_dim_float = float(self.dk_dim)
        scaled_sim_bnt = sim_bnt / math.sqrt(dk_dim_float)

        # 時間方向 Softmax (重み $\alpha$)
        alpha = F.softmax(scaled_sim_bnt, dim=2) # T次元(フレーム数)でSoftmax $\alpha$: (B, N, T)

        # 5-3. 重み付き集約 (数式 17): $v_p$ の計算
        v_p = torch.bmm(alpha, V_v) # v_p: (B, N, D)

        # 5-4. Transformerスタイルの残差ブロック (数式 18, 19): $v'$ の計算
        A_prime_Ws = self.Ws(fused_action_vectors) # (B, N, D)
        
        # 数式 (18): $\tilde{v} = \mathrm{LN}(v_p + A'W_s)$
        v_tilde = self.norm_1(v_p + A_prime_Ws) 
        
        # 数式 (19): $v' = \mathrm{LN}(\tilde{v} + \mathrm{FFN}(\tilde{v}))$
        v_prime = self.norm_2(v_tilde + self.ffn(v_tilde)) # $v'$: (B, N, D)
        
        final_video_vectors = v_prime
        
        # --- 6) 最終コサイン類似度 (ロジットスケールなし) ---
        eps = 1e-6
        
        # $v'$ を正規化
        pooled_norm = final_video_vectors / (final_video_vectors.norm(dim=-1, keepdim=True) + eps) # (B, N, D)
        
        # $A'$ を正規化
        fused_norm = fused_action_vectors / (fused_action_vectors.norm(dim=-1, keepdim=True) + eps) # (B, N, D)

        # コサイン類似度を直接ロジットとする
        logits = (pooled_norm * fused_norm).sum(dim=-1)  # (B, N)
        
        return logits, video_vector


def build_vcw_model(
    state_dict: dict,
    T: int,
    temporal_layers: int,
    temporal_heads: int,
    device: Union[str, torch.device] = "cuda",
    use_logit_scale: bool = True,
    # 学習済みアダプタを適用するための引数
    external_adapter_dir: str = "",
    external_state_filename: str = "global_adapter_state.pt",
    external_adapter_trainable: bool = False,
):
    """
    OpenAI CLIPのstate_dictからVideoChangeWeightedCLIPモデルを構築する。
    external_adapter_dir が指定され、かつ global_adapter_state.pt が存在する場合は
    CLIP-Adapter を (1-a):a でフレーム特徴にブレンドして使用する。
    """
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    # external_adapter_dir は存在チェックした上で渡す
    use_external_dir = ""
    if external_adapter_dir:
        candidate = os.path.join(external_adapter_dir, external_state_filename)
        if os.path.isfile(candidate):
            use_external_dir = external_adapter_dir
        else:
            print(f"[VCW-CLIP] global_adapter_state not found at: {candidate}. Proceeding without adapter.")

    model = VideoChangeWeightedCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        T=T,
        temporal_layers=temporal_layers,
        temporal_heads=temporal_heads,
        device=device,
        use_logit_scale=use_logit_scale,
        external_adapter_dir=use_external_dir,
        external_state_filename=external_state_filename,
        external_adapter_trainable=external_adapter_trainable,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print("Keys not loaded (expected for VCW-CLIP):", msg.missing_keys)
    return model.eval()

def vcw_clip_load(
    config: "yacs.config.CfgNode",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    VCW-CLIPモデルをロードする。
    - HF_FINETUNED_PATH が指定されていれば HF 形式の CLIP を基に VCW-CLIP を構築。
      同ディレクトリに global_adapter_state.pt がある場合はフレーム用 CLIP-Adapter を適用。
    - 指定が無い場合は標準の CLIP 重みを基に VCW-CLIP を構築（アダプタ無し）。
    """
    hf_path = config.MODEL.HF_FINETUNED_PATH
    expected_arch = config.MODEL.ARCH

    if hf_path:
        print(f"--- Loading Fine-Tuned HF Model from {hf_path} for VCW-CLIP ---")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ft_model = CLIPModel.from_pretrained(hf_path).to("cpu")

        hf_state_dict = ft_model.state_dict()
        hf_config = ft_model.config
        vision_layers = hf_config.vision_config.num_hidden_layers
        text_layers = hf_config.text_config.num_hidden_layers

        expected_layers = 12 if expected_arch in ("ViT-B/32", "ViT-B/16") else 0
        if expected_layers == 0 or vision_layers != expected_layers or text_layers != expected_layers:
            raise ValueError(
                f"Architecture mismatch: Config '{expected_arch}' expects {expected_layers} layers, "
                f"but HF model has vision={vision_layers}, text={text_layers}."
            )
        del ft_model

        print("--- Converting HF state_dict to OpenAI format ---")
        openai_state_dict = {}
        t = "text_model."
        v = "vision_model."

        # Text
        openai_state_dict["token_embedding.weight"] = hf_state_dict[t + "embeddings.token_embedding.weight"]
        openai_state_dict["positional_embedding"] = hf_state_dict[t + "embeddings.position_embedding.weight"]
        for i in range(text_layers):
            q_w = hf_state_dict[f"{t}encoder.layers.{i}.self_attn.q_proj.weight"]
            k_w = hf_state_dict[f"{t}encoder.layers.{i}.self_attn.k_proj.weight"]
            v_w = hf_state_dict[f"{t}encoder.layers.{i}.self_attn.v_proj.weight"]
            openai_state_dict[f"transformer.resblocks.{i}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf_state_dict[f"{t}encoder.layers.{i}.self_attn.q_proj.bias"]
            k_b = hf_state_dict[f"{t}encoder.layers.{i}.self_attn.k_proj.bias"]
            v_b = hf_state_dict[f"{t}encoder.layers.{i}.self_attn.v_proj.bias"]
            openai_state_dict[f"transformer.resblocks.{i}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)

            openai_state_dict[f"transformer.resblocks.{i}.attn.out_proj.weight"] = hf_state_dict[f"{t}encoder.layers.{i}.self_attn.out_proj.weight"]
            openai_state_dict[f"transformer.resblocks.{i}.attn.out_proj.bias"] = hf_state_dict[f"{t}encoder.layers.{i}.self_attn.out_proj.bias"]
            openai_state_dict[f"transformer.resblocks.{i}.ln_1.weight"] = hf_state_dict[f"{t}encoder.layers.{i}.layer_norm1.weight"]
            openai_state_dict[f"transformer.resblocks.{i}.ln_1.bias"] = hf_state_dict[f"{t}encoder.layers.{i}.layer_norm1.bias"]
            openai_state_dict[f"transformer.resblocks.{i}.mlp.c_fc.weight"] = hf_state_dict[f"{t}encoder.layers.{i}.mlp.fc1.weight"]
            openai_state_dict[f"transformer.resblocks.{i}.mlp.c_fc.bias"] = hf_state_dict[f"{t}encoder.layers.{i}.mlp.fc1.bias"]
            openai_state_dict[f"transformer.resblocks.{i}.mlp.c_proj.weight"] = hf_state_dict[f"{t}encoder.layers.{i}.mlp.fc2.weight"]
            openai_state_dict[f"transformer.resblocks.{i}.mlp.c_proj.bias"] = hf_state_dict[f"{t}encoder.layers.{i}.mlp.fc2.bias"]
            openai_state_dict[f"transformer.resblocks.{i}.ln_2.weight"] = hf_state_dict[f"{t}encoder.layers.{i}.layer_norm2.weight"]
            openai_state_dict[f"transformer.resblocks.{i}.ln_2.bias"] = hf_state_dict[f"{t}encoder.layers.{i}.layer_norm2.bias"]

        openai_state_dict["ln_final.weight"] = hf_state_dict[t + "final_layer_norm.weight"]
        openai_state_dict["ln_final.bias"] = hf_state_dict[t + "final_layer_norm.bias"]

        # Vision
        openai_state_dict["visual.conv1.weight"] = hf_state_dict[v + "embeddings.patch_embedding.weight"]
        openai_state_dict["visual.class_embedding"] = hf_state_dict[v + "embeddings.class_embedding"]
        openai_state_dict["visual.positional_embedding"] = hf_state_dict[v + "embeddings.position_embedding.weight"]
        openai_state_dict["visual.ln_pre.weight"] = hf_state_dict[v + "pre_layrnorm.weight"]
        openai_state_dict["visual.ln_pre.bias"] = hf_state_dict[v + "pre_layrnorm.bias"]
        for i in range(vision_layers):
            q_w = hf_state_dict[f"{v}encoder.layers.{i}.self_attn.q_proj.weight"]
            k_w = hf_state_dict[f"{v}encoder.layers.{i}.self_attn.k_proj.weight"]
            v_w = hf_state_dict[f"{v}encoder.layers.{i}.self_attn.v_proj.weight"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)

            q_b = hf_state_dict[f"{v}encoder.layers.{i}.self_attn.q_proj.bias"]
            k_b = hf_state_dict[f"{v}encoder.layers.{i}.self_attn.k_proj.bias"]
            v_b = hf_state_dict[f"{v}encoder.layers.{i}.self_attn.v_proj.bias"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)

            openai_state_dict[f"visual.transformer.resblocks.{i}.attn.out_proj.weight"] = hf_state_dict[f"{v}encoder.layers.{i}.self_attn.out_proj.weight"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.attn.out_proj.bias"] = hf_state_dict[f"{v}encoder.layers.{i}.self_attn.out_proj.bias"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.ln_1.weight"] = hf_state_dict[f"{v}encoder.layers.{i}.layer_norm1.weight"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.ln_1.bias"] = hf_state_dict[f"{v}encoder.layers.{i}.layer_norm1.bias"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_fc.weight"] = hf_state_dict[f"{v}encoder.layers.{i}.mlp.fc1.weight"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_fc.bias"] = hf_state_dict[f"{v}encoder.layers.{i}.mlp.fc1.bias"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_proj.weight"] = hf_state_dict[f"{v}encoder.layers.{i}.mlp.fc2.weight"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_proj.bias"] = hf_state_dict[f"{v}encoder.layers.{i}.mlp.fc2.bias"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.ln_2.weight"] = hf_state_dict[f"{v}encoder.layers.{i}.layer_norm2.weight"]
            openai_state_dict[f"visual.transformer.resblocks.{i}.ln_2.bias"] = hf_state_dict[f"{v}encoder.layers.{i}.layer_norm2.bias"]

        openai_state_dict["visual.ln_post.weight"] = hf_state_dict[v + "post_layernorm.weight"]
        openai_state_dict["visual.ln_post.bias"] = hf_state_dict[v + "post_layernorm.bias"]

        # Projections
        openai_state_dict["text_projection"] = hf_state_dict["text_projection.weight"].T
        openai_state_dict["visual.proj"] = hf_state_dict["visual_projection.weight"].T
        openai_state_dict["logit_scale"] = hf_state_dict["logit_scale"]

        print("Conversion complete.")

        # HF パスにアダプタ state があれば CLIP-Adapter を適用
        model = build_vcw_model(
            state_dict=openai_state_dict,
            T=config.DATA.NUM_FRAMES,
            temporal_layers=config.MODEL.VCW_TEMPORAL_LAYERS,
            temporal_heads=config.MODEL.VCW_TEMPORAL_HEADS,
            device=device,
            use_logit_scale=True,
            external_adapter_dir=hf_path,  # build_vcw_model 内で存在チェック
            external_state_filename="global_adapter_state.pt",
            external_adapter_trainable=False,  # 微調整したければ True
        )

    else:
        # 標準の CLIP 重み
        model_path = config.MODEL.PRETRAINED
        name = config.MODEL.ARCH

        if model_path is None:
            model_path = clip._download(clip._MODELS[name])

        try:
            jit_model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = jit_model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path} (or failed to download {name})")

        # ピュア CLIP 挙動（logit_scale 無効化, アダプタ無し）
        model = build_vcw_model(
            state_dict=state_dict,
            T=config.DATA.NUM_FRAMES,
            temporal_layers=config.MODEL.VCW_TEMPORAL_LAYERS,
            temporal_heads=config.MODEL.VCW_TEMPORAL_HEADS,
            device=device,
            use_logit_scale=False,
            external_adapter_dir="",  # アダプタ無し
        )

    if str(device) == "cpu":
        model.float()
    return model


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_size, output_size))
        self.leakrelu = nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(output_size)
        self.input_size = input_size
        self.output_size = output_size

    def edges2matrix(self, edges, num1, num2):
        num = num1 + num2
        matrix = torch.zeros(num, num).cuda() 
        
        for i in edges:
            idx0 = i[0] - 1
            idx1 = i[1] - 1
            
            if (0 <= idx0 < num) and (0 <= idx1 < num):
                matrix[idx0, idx1] = 1
            
        return matrix

    def forward(self, x, edges, num1, num2):
        b = x.shape[0]
        adjacency_matrix = self.edges2matrix(edges, num1, num2)  # 根据边的连接关系设置邻接矩阵
        adjacency_matrix = adjacency_matrix.expand(b, -1, -1)

        # 计算邻接矩阵和权重的乘积
        x = torch.matmul(adjacency_matrix.to(x.dtype), torch.matmul(x, self.weight))
        if self.output_size <= self.input_size:
            x = self.leakrelu(x)
        x = self.norm(x)
        return x

class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(GraphConvolution(input_size, hidden_size))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_size, hidden_size))

        # 输出层
        self.layers.append(GraphConvolution(hidden_size, output_size))

    def forward(self, x, edges, num1, num2):
        for layer in self.layers:
            x = layer(x, edges, num1, num2)
        return x

class XCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 # video
                 T=8, 
                 droppath=0.,
                 mit_layers=1,
                 # prompt 
                 prompts_alpha=1e-4,
                 prompts_layers=3,
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        
        self.prompts_generator1 = AnimalSpecificPrompt(layers=prompts_layers, embed_dim=embed_dim, alpha=prompts_alpha,)
        self.prompts_generator2 = AnimalSpecificPrompt(layers=prompts_layers, embed_dim=embed_dim, alpha=prompts_alpha,)
        self.use_cache=use_cache
        self.mit = MultiframeIntegrationTransformer(T=T, embed_dim=embed_dim, layers=mit_layers,)

        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        self.visual = CrossFrameCommunicationTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None
        self.cache_animal_text_features = None
        self.prompts_visual_ln = LayerNorm(vision_width)
        self.prompts_visual_proj = nn.Parameter(torch.randn(vision_width, embed_dim))
        
        self.graph = GCNModel(embed_dim, 256, embed_dim, 2)
        
        self.initialize_parameters()
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        # テンソルが long (int64) 型であることを保証する
        text = text.to(torch.long)
        x = self.token_embedding(text)
        
        # <EOS> トークンIDが最大値であるという前提のロジック
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape
        
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # [K, C] の形状を抽出
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        
        # 元のコードにあった x.reshape(K, -1) は、
        # x が既に [K, C] であるため不要であり、削除した。
        return x

    def encode_video(self, image, name=''):
        b,t,c,h,w = image.size()
        image = image.reshape(-1,c,h,w)

        cls_features, img_features = self.encode_image(image)
        img_features = self.prompts_visual_ln(img_features)
        img_features = img_features @ self.prompts_visual_proj
        
        cls_features = cls_features.view(b, t, -1)
        img_features = img_features.view(b,t,-1,cls_features.shape[-1])
        video_features = self.mit(cls_features)

        return video_features, img_features
    
    def cache_text(self, text):
        """
        テキスト特徴量を計算しキャッシュする。
        入力 text の形状が (N, K, 77) の場合、(N, K, 512) にエンコード後、
        (N, 512) に集約 (max-pooling) する。
        形状が (N, 77) の場合は、(N, 512) にエンコードするのみ。
        """
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                
                # 入力の次元数に基づいて、クラス数とプロンプト数を決定する
                if text.dim() == 3:
                    # 入力形状: (N, K, 77)
                    n_classes, n_prompts, _ = text.shape
                    # (N, K, 77) -> (N * K, 77)
                    text_input = text.view(n_classes * n_prompts, -1)
                else:
                    # 入力形状: (N, 77)
                    n_classes = text.shape[0]
                    n_prompts = 1
                    text_input = text
                
                total_prompts = text_input.shape[0]

                # DataLoader は (N * K, 77) または (N, 77) に対して動作
                data_loader = DataLoader(text_input, batch_size=50, shuffle=False)
                
                # embed_dim は 512 と仮定
                text_features = torch.zeros([total_prompts, 512]).cuda()
                start_index = 0
                for data in data_loader:
                    text_feature = self.encode_text(data)
                    end_index = int(start_index + text_feature.size(0))
                    text_features[start_index:end_index] = text_feature
                    start_index = end_index
                
                # 1クラス1プロンプトより多い場合、集約処理を行う
                if n_prompts > 1:
                    # (N * K, 512) -> (N, K, 512)
                    text_features = text_features.view(n_classes, n_prompts, 512)
                    # (N, K, 512) -> (N, 512) (K次元で Max-Pooling)
                    text_features, _ = torch.max(text_features, dim=1)
                
                # n_prompts = 1 の場合、
                # text_features は (N, 512) のまま変更されない。
                
                print('text_features', text_features.shape)
                self.cache_text_features = text_features
        self.train()
        return self.cache_text_features

    def cache_animal_text(self, text):
        """
        animal_text 特徴量を計算しキャッシュする。
        cache_text と全く同じロジックで動作する。
        """
        self.eval()
        with torch.no_grad():
            if self.cache_animal_text_features is None:
                
                if text.dim() == 3:
                    # (N, K, 77)
                    n_classes, n_prompts, _ = text.shape
                    text_input = text.view(n_classes * n_prompts, -1)
                else:
                    # (N, 77)
                    n_classes = text.shape[0]
                    n_prompts = 1
                    text_input = text
                
                total_prompts = text_input.shape[0]

                data_loader = DataLoader(text_input, batch_size=50, shuffle=False)
                text_features = torch.zeros([total_prompts, 512]).cuda()
                start_index = 0
                for data in data_loader:
                    text_feature = self.encode_text(data)
                    end_index = int(start_index + text_feature.size(0))
                    text_features[start_index:end_index] = text_feature
                    start_index = end_index

                if n_prompts > 1:
                    # (N * K, 512) -> (N, K, 512)
                    text_features = text_features.view(n_classes, n_prompts, 512)
                    # (N, K, 512) -> (N, 512)
                    text_features, _ = torch.max(text_features, dim=1)
                
                print('animal_text_features', text_features.shape)
                self.cache_animal_text_features = text_features
        self.train()
        return self.cache_animal_text_features

    def cache_animal_logits(self, text, image_features, animal_text_features):
        self.eval()
        with torch.no_grad():
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            animal_text_features = animal_text_features / animal_text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            animal_logits = torch.einsum("bd,kd->bk", image_features, logit_scale * animal_text_features).softmax(dim=-1)
        self.train()
        return animal_logits

    def forward(self, image, text, animal_text, animal_pred, edges, name='', pred=False):
        b = image.shape[0]
        T = image.shape[1]
        video_features, img_features = self.encode_video(image, name)
        
        if text.dim() == 3:
            num2 = text.shape[0]  # クラス数 N
        else:
            num2 = text.shape[0]  # クラス数 N
            
        if animal_text.dim() == 3:
            num1 = animal_text.shape[0] # クラス数 N
        else:
            num1 = animal_text.shape[0] # クラス数 N
        
        logit_scale = self.logit_scale.exp()
        
        if self.use_cache:
            # cache_text / cache_animal_text は、入力が 3D (N, K, 77) でも 
            # 2D (N, 77) でも、常に 2D (N, 512) を返すように修正された。
            # したがって、forward 側での 3D -> 2D への view 処理は不要である。
            text_features = self.cache_text(text)
            animal_text_features = self.cache_animal_text(animal_text)
        else:            
            # (1) animal_text の処理
            if animal_text.dim() == 3:
                n_classes, n_prompts, _ = animal_text.shape
                animal_text_input = animal_text.view(n_classes * n_prompts, -1)
            else:
                n_classes = animal_text.shape[0]
                n_prompts = 1
                animal_text_input = animal_text

            animal_text_features = self.encode_text(animal_text_input)
            
            if n_prompts > 1:
                animal_text_features = animal_text_features.view(n_classes, n_prompts, 512)
                animal_text_features, _ = torch.max(animal_text_features, dim=1)
            
            # (2) text の処理
            if text.dim() == 3:
                n_classes, n_prompts, _ = text.shape
                text_input = text.view(n_classes * n_prompts, -1)
            else:
                n_classes = text.shape[0]
                n_prompts = 1
                text_input = text
            
            text_features = self.encode_text(text_input)

            if n_prompts > 1:
                text_features = text_features.view(n_classes, n_prompts, 512)
                text_features, _ = torch.max(text_features, dim=1)

        # [num1, 512] -> [b, num1, 512]
        animal_text_features = animal_text_features.unsqueeze(0).expand(b, -1, -1) 
        # [num2, 512] -> [b, num2, 512]
        text_features = text_features.unsqueeze(0).expand(b, -1, -1) 
        
        animal_pred = animal_pred.unsqueeze(1) # [b, 1, 512]
        
        animal_text_feature = torch.einsum('bac, bcm->bam', [animal_pred.to(animal_text_features.dtype), animal_text_features])
        
        animal_pred = animal_pred.squeeze(1).unsqueeze(-1) # [b, 512, 1]
        
        animal_text_features = torch.mul(animal_pred.to(animal_text_features.dtype), animal_text_features)
        
        edges = edges.clone().detach()

        x = torch.cat([animal_text_features, text_features], dim=1) # [b, num1 + num2, 512]
        text_features = self.graph(x, edges, num1, num2)[:, animal_text_features.shape[1]:, :] # [b, num2, 512]
        
        text_features = text_features + self.prompts_generator1(text_features, animal_text_feature)
        
        video_features = video_features.unsqueeze(1)
        video_features = video_features + self.prompts_generator2(video_features, animal_text_feature)
        video_features = video_features.squeeze(1)
        
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)
        return logits, video_features


def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None, prompts_alpha=1e-1, prompts_layers=2, use_cache=True, mit_layers=4, pred=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    
    model = XCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,  
        T=T, droppath=droppath, mit_layers=mit_layers,
        prompts_alpha=prompts_alpha, prompts_layers=prompts_layers,
        use_checkpoint=use_checkpoint, use_cache=use_cache,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict,strict=False)

    return model.eval()


def xclip_load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", 
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, 
         use_cache=True, pred=False, prompts_alpha=1e-1, prompts_layers=3, mit_layers=1,
):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath, 
                        use_checkpoint=use_checkpoint, 
                        logger=logger,
                        prompts_alpha=prompts_alpha, 
                        prompts_layers=prompts_layers,
                        use_cache=use_cache,
                        mit_layers=mit_layers,
                        pred=pred,
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()


def load_finetuned_xclip_model(
    hf_model_path: str, 
    device: str,
    xclip_params: dict,
    expected_arch: str # config.MODEL.ARCH を受け取る
) -> XCLIP:
    """
    Hugging Face形式のファインチューニング済みCLIPモデルをロードし、
    OpenAI形式に変換してXCLIPモデルに読み込ませる。
    
    エラー抑制(try-except)は行わず、問題発生時にはエラーを送出する。
    """
    
    print(f"--- Loading Fine-Tuned HF Model from {hf_model_path} ---")
    
    # ステップ1: HF形式のモデルをロード
    # OSErrorが発生した場合、ここで停止する
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # config.json の警告を抑制
        ft_model = CLIPModel.from_pretrained(hf_model_path).to(device)
    
    hf_state_dict = ft_model.state_dict()
    
    # --- アーキテクチャ検証 ---
    config = ft_model.config
    vision_layers = config.vision_config.num_hidden_layers
    text_layers = config.text_config.num_hidden_layers
    
    # ViT-B (12層) の変換ロジックを前提としているため、レイヤー数を確認
    expected_layers = 0
    if expected_arch == 'ViT-B/32' or expected_arch == 'ViT-B/16':
        expected_layers = 12
    # elif expected_arch == 'ViT-L/14':
    #     expected_layers = 24 # 注: 変換ロジックが ViT-L に未対応
        
    if expected_layers == 0:
        raise ValueError(
            f"Unsupported expected_arch: '{expected_arch}'. "
            "Conversion logic only supports 'ViT-B/16' or 'ViT-B/32'."
        )

    if (vision_layers != expected_layers or text_layers != expected_layers):
        raise ValueError(
            f"Architecture mismatch: Config specifies '{expected_arch}' ({expected_layers} layers), "
            f"but HF model has {vision_layers} vision layers "
            f"and {text_layers} text layers. "
            "Conversion logic will fail."
        )

    del ft_model # メモリ解放

    print("--- Converting HF state_dict to OpenAI format ---")
    
    # ステップ2: HF state_dict を OpenAI state_dict に変換
    # KeyError が発生した場合、ここで停止する
    openai_state_dict = {}
    hf_key_prefix_text = "text_model."
    hf_key_prefix_vision = "vision_model."

    # --- Text Model 変換 ---
    openai_state_dict["token_embedding.weight"] = hf_state_dict[hf_key_prefix_text + "embeddings.token_embedding.weight"]
    openai_state_dict["positional_embedding"] = hf_state_dict[hf_key_prefix_text + "embeddings.position_embedding.weight"]
    
    for i in range(text_layers):
        q_w = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.self_attn.q_proj.weight"]
        k_w = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.self_attn.k_proj.weight"]
        v_w = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.self_attn.v_proj.weight"]
        openai_state_dict[f"transformer.resblocks.{i}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        
        q_b = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.self_attn.q_proj.bias"]
        k_b = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.self_attn.k_proj.bias"]
        v_b = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.self_attn.v_proj.bias"]
        openai_state_dict[f"transformer.resblocks.{i}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        openai_state_dict[f"transformer.resblocks.{i}.attn.out_proj.weight"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.self_attn.out_proj.weight"]
        openai_state_dict[f"transformer.resblocks.{i}.attn.out_proj.bias"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.self_attn.out_proj.bias"]
        openai_state_dict[f"transformer.resblocks.{i}.ln_1.weight"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.layer_norm1.weight"]
        openai_state_dict[f"transformer.resblocks.{i}.ln_1.bias"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.layer_norm1.bias"]
        openai_state_dict[f"transformer.resblocks.{i}.mlp.c_fc.weight"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.mlp.fc1.weight"]
        openai_state_dict[f"transformer.resblocks.{i}.mlp.c_fc.bias"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.mlp.fc1.bias"]
        openai_state_dict[f"transformer.resblocks.{i}.mlp.c_proj.weight"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.mlp.fc2.weight"]
        openai_state_dict[f"transformer.resblocks.{i}.mlp.c_proj.bias"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.mlp.fc2.bias"]
        openai_state_dict[f"transformer.resblocks.{i}.ln_2.weight"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.layer_norm2.weight"]
        openai_state_dict[f"transformer.resblocks.{i}.ln_2.bias"] = hf_state_dict[f"{hf_key_prefix_text}encoder.layers.{i}.layer_norm2.bias"]

    openai_state_dict["ln_final.weight"] = hf_state_dict[hf_key_prefix_text + "final_layer_norm.weight"]
    openai_state_dict["ln_final.bias"] = hf_state_dict[hf_key_prefix_text + "final_layer_norm.bias"]

    # --- Vision Model 変換 (ViT-B) ---
    openai_state_dict["visual.conv1.weight"] = hf_state_dict[hf_key_prefix_vision + "embeddings.patch_embedding.weight"]
    
    openai_state_dict["visual.class_embedding"] = hf_state_dict[hf_key_prefix_vision + "embeddings.class_embedding"]
    openai_state_dict["visual.positional_embedding"] = hf_state_dict[hf_key_prefix_vision + "embeddings.position_embedding.weight"]
    
    openai_state_dict["visual.ln_pre.weight"] = hf_state_dict[hf_key_prefix_vision + "pre_layrnorm.weight"]
    openai_state_dict["visual.ln_pre.bias"] = hf_state_dict[hf_key_prefix_vision + "pre_layrnorm.bias"]

    for i in range(vision_layers):
        q_w = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.self_attn.q_proj.weight"]
        k_w = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.self_attn.k_proj.weight"]
        v_w = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.self_attn.v_proj.weight"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        
        q_b = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.self_attn.q_proj.bias"]
        k_b = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.self_attn.k_proj.bias"]
        v_b = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.self_attn.v_proj.bias"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        
        openai_state_dict[f"visual.transformer.resblocks.{i}.attn.out_proj.weight"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.self_attn.out_proj.weight"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.attn.out_proj.bias"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.self_attn.out_proj.bias"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.ln_1.weight"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.layer_norm1.weight"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.ln_1.bias"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.layer_norm1.bias"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_fc.weight"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.mlp.fc1.weight"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_fc.bias"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.mlp.fc1.bias"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_proj.weight"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.mlp.fc2.weight"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_proj.bias"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.mlp.fc2.bias"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.ln_2.weight"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.layer_norm2.weight"]
        openai_state_dict[f"visual.transformer.resblocks.{i}.ln_2.bias"] = hf_state_dict[f"{hf_key_prefix_vision}encoder.layers.{i}.layer_norm2.bias"]

    openai_state_dict["visual.ln_post.weight"] = hf_state_dict[hf_key_prefix_vision + "post_layernorm.weight"]
    openai_state_dict["visual.ln_post.bias"] = hf_state_dict[hf_key_prefix_vision + "post_layernorm.bias"]

    # --- Projections 変換 ---
    openai_state_dict["text_projection"] = hf_state_dict["text_projection.weight"].T
    openai_state_dict["visual.proj"] = hf_state_dict["visual_projection.weight"].T
    
    openai_state_dict["logit_scale"] = hf_state_dict["logit_scale"]

    print("Conversion complete.")

    # ステップ3: build_model を使用して XCLIP モデルを構築・ロード
    print("--- Building XCLIP model with converted state_dict ---")
    
    # build_model は渡された state_dict からアーキテクチャを推論する
    model = build_model(
        openai_state_dict, 
        **xclip_params
    )
    model = model.to(device)

    print("XCLIP model loaded successfully with fine-tuned weights.")
    return model


_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "2d": [[1, 1, 1]],
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "x3d": [[1, 1, 1]],
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]

class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.DATA.NUM_CLASSES,
            pool_size=[None, None]
            if cfg.MULTIGRID.SHORT_CYCLE
            else [
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[1][2],
                ],
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)
        return x

class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )
        
        self.head = head_helper.ResNetBasicHead(
            dim_in=[width_per_group * 32],
            num_classes=cfg.DATA.NUM_CLASSES,
            pool_size=[None, None]
            if cfg.MULTIGRID.SHORT_CYCLE
            else [
                [
                    cfg.DATA.NUM_FRAMES // pool_size[0][0],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.INPUT_SIZE // 32 // pool_size[0][2],
                ]
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x)
        return x

class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.

    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 1

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner]
                if cfg.X3D.CHANNELWISE_3x3x3
                else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        spat_sz = int(math.ceil(cfg.DATA.INPUT_SIZE / 32.0))
        self.head = head_helper.X3DHead(
            dim_in=dim_out,
            dim_inner=dim_inner,
            dim_out=cfg.X3D.DIM_C5,
            num_classes=cfg.DATA.NUM_CLASSES,
            pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            bn_lin5_on=cfg.X3D.BN_LIN5,
        )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x

class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        # assert cfg.DATA.INPUT_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        # Prepare input.
        spatial_size = cfg.DATA.INPUT_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = cfg.DATA.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        # num_patches = math.prod(self.patch_dims)
        num_patches = reduce(operator.mul, self.patch_dims, 1)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        pool_q = cfg.MVIT.POOL_Q_KERNEL
        pool_kv = cfg.MVIT.POOL_KV_KERNEL
        pool_skip = cfg.MVIT.POOL_SKIP_KERNEL
        stride_q = cfg.MVIT.POOL_Q_STRIDE
        stride_kv = cfg.MVIT.POOL_KV_STRIDE
        stride_skip = cfg.MVIT.POOL_SKIP_STRIDE

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)

        if len(cfg.MVIT.DIM_MUL) > 1:
            for k in cfg.MVIT.DIM_MUL:
                dim_mul[k[0]] = k[1]
        if len(cfg.MVIT.HEAD_MUL) > 1:
            for k in cfg.MVIT.HEAD_MUL:
                head_mul[k[0]] = k[1]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )

            self.blocks.append(
                MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    kernel_skip=pool_skip[i] if len(pool_skip) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    stride_skip=stride_skip[i] if len(stride_skip) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                )
            )

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        self.head = head_helper.TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x):
        x = x[0].squeeze(2)
        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.INPUT_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.INPUT_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            pos_embed_cls = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed_cls
        else:
            x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, 0]
        else:
            x = x.mean(1)

        x = self.head(x)
        return x

class TemporalModelling(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, dropout: float, attn_mask: torch.Tensor = None, ):
        super(TemporalModelling, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dropout ) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class VideoPrompt(torch.nn.Module):
    def __init__(self, config, actionlist, actiondict, actiontoken, device='cuda'):
        super(VideoPrompt, self).__init__()

        self.device = device
        self.clipmodel, _ = clip.load(config.MODEL.ARCH, device=self.device, jit=False)

        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False

        self.dropout = 0.0 # if args.tfm_layers > 2 else 0.0
        self.hidden_size = 512
        self.numF = config.DATA.NUM_FRAMES

        self.prefix = 16
        self.postfix = 16
        self.actionlist = actionlist
        self.actiondict = actiondict
        self.actiontoken = actiontoken
        self.tfm_layers = 1
        self.tfm_heads = 8

        self.embedding = torch.nn.Embedding(77, self.hidden_size)
        self.temporalEmbedding = torch.nn.Embedding(self.numF, self.hidden_size)

        self.temporalModelling = TemporalModelling(width=self.hidden_size, layers=self.tfm_layers, heads=self.tfm_heads, dropout=self.dropout)

        self.initialize_parameters()


    def initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.temporalEmbedding.weight, std=0.01)


    def replace_text_embedding(self, actionlist):
        self.text_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat([len(actionlist), 1, 1])
        self.prompt_actiontoken = torch.zeros(len(actionlist), 77)
        for i, a in enumerate(actionlist):
            embedding = torch.from_numpy(self.actiondict[a][0]).float().to(self.device)
            token = torch.from_numpy(self.actiontoken[a][0])
            self.text_embedding[i][0] = embedding[0]
            ind = np.argmax(token, -1)

            self.text_embedding[i][self.prefix + 1: self.prefix + ind] = embedding[1:ind]
            self.text_embedding[i][self.prefix + ind + self.postfix] = embedding[ind]

            self.prompt_actiontoken[i][0] = token[0]
            self.prompt_actiontoken[i][self.prefix + 1: self.prefix + ind] = token[1:ind]
            self.prompt_actiontoken[i][self.prefix + ind + self.postfix] = token[ind]

        self.text_embedding.to(self.device)
        self.prompt_actiontoken.to(self.device)


    def forward(self, vids, inp_actionlist):
        # replace_text_embedding at every iter
        # otherwise RuntimeError: backward through the graph a second time
        self.replace_text_embedding(inp_actionlist)

        # encode text
        tFeature = self.clipmodel.encode_text(self.prompt_actiontoken, self.text_embedding)

        # encode videos
        # vFeature = einops.rearrange(vids.float(), 'b t c -> t b c', t=self.numF)
        
        iFeature = self.clipmodel.encode_image(einops.rearrange(vids, 'b t c h w -> (b t) c h w'))
        vFeature = einops.rearrange(iFeature, '(b t) c -> t b c', t=self.numF)

        # temporal modelling
        tempEmbedding = einops.repeat(self.temporalEmbedding(torch.arange(self.numF).to(self.device)), 't c -> t b c', b=vFeature.size(1))
        vFeature = vFeature + tempEmbedding.to(self.device)
        vFeature = self.temporalModelling(vFeature)  
        vFeature = vFeature.mean(dim=0)
        
        vFeature = vFeature / vFeature.norm(dim=-1, keepdim=True)
        tFeature = tFeature / tFeature.norm(dim=-1, keepdim=True)
        logits = vFeature @ tFeature.t() / 0.07

        return logits

class EVLDecoder(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        spatial_size: Tuple[int, int] = (14, 14),
        num_layers: int = 4,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in range(num_layers)]
        )

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList(
                [nn.Conv1d(in_feature_dim, in_feature_dim, kernel_size=3, stride=1, padding=1, groups=in_feature_dim) for _ in range(num_layers)]
            )
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(num_layers)]
            )
        if enable_temporal_cross_attention:
            self.cross_attention = nn.ModuleList(
                [TemporalCrossAttention(spatial_size, in_feature_dim) for _ in range(num_layers)]
            )

        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


    def forward(self, in_features: List[Dict[str, torch.Tensor]]):
        N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)

        for i in range(self.num_layers):
            frame_features = in_features[i]['out']
            
            if self.enable_temporal_conv:
                feat = in_features[i]['out']
                feat = feat.permute(0, 2, 3, 1).contiguous().flatten(0, 1) # N * L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C, T).permute(0, 3, 1, 2).contiguous() # N, T, L, C
                frame_features += feat
            
            if self.enable_temporal_pos_embed:
                frame_features += self.temporal_pos_embed[i].view(1, T, 1, C)
            
            if self.enable_temporal_cross_attention:
                frame_features += self.cross_attention[i](in_features[i]['q'], in_features[i]['k'])

            frame_features = frame_features.flatten(1, 2) # N, T * L, C
            
            x = self.decoder_layers[i](x, frame_features)
        
        return x


class EVLTransformer(nn.Module):

    def __init__(
        self, config,
        backbone_type: str = 'clip',
        backbone_path: str = '~/.cache/clip/ViT-B-16.pt',
        backbone_mode: str = 'freeze_fp16',
        decoder_num_layers: int = 4,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        cls_dropout: float = 0.5,
        decoder_mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers

        backbone_config = self._create_backbone(config, backbone_type, backbone_path, backbone_mode)
        backbone_feature_dim = backbone_config['feature_dim']
        backbone_spatial_size = tuple(x // y for x, y in zip(backbone_config['input_size'], backbone_config['patch_size']))

        self.decoder = EVLDecoder(
            num_frames=config.DATA.NUM_FRAMES,
            spatial_size=backbone_spatial_size,
            num_layers=decoder_num_layers,
            in_feature_dim=backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            enable_temporal_conv=enable_temporal_conv,
            enable_temporal_pos_embed=enable_temporal_pos_embed,
            enable_temporal_cross_attention=enable_temporal_cross_attention,
            mlp_dropout=decoder_mlp_dropout,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(backbone_feature_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(backbone_feature_dim, config.DATA.NUM_CLASSES),
        )


    def _create_backbone(
        self, config,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = VisionTransformer2D(return_all_features=True, **vit_presets[config.MODEL.ARCH])
        backbone.load_state_dict(state_dict, strict=True) # weight_loader_fn is expected to strip unused parameters

        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']

        if backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval().requires_grad_(False)
            if backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)
            self.backbone = [backbone] # avoid backbone parameter registration

        return vit_presets[config.MODEL.ARCH]


    def _get_backbone(self, x):
        if isinstance(self.backbone, list):
            # freeze backbone
            self.backbone[0] = self.backbone[0].to(x.device)
            return self.backbone[0]
        else:
            # finetune bakbone
            return self.backbone


    def forward(self, x: torch.Tensor):
        x = x[0]
        backbone = self._get_backbone(x)

        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        features = backbone(x)[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        x = self.decoder(features)
        
        vfeature = x.mean(0)
        
        x = self.proj(x[:, 0, :])

        return x #, vfeature
    
class TimeSformerCLIPInitVideoGuide(nn.Module):
    def __init__(self, class_embed, num_frames):
        super().__init__()
        self.num_classes, self.embed_dim = class_embed.shape
        self.backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", num_frames=num_frames, ignore_mismatched_sizes=True)
        self.linear1 = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=self.embed_dim, bias=False)
        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)
        self.image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.linear2 = nn.Linear(in_features=self.backbone.config.hidden_size + self.embed_dim, out_features=self.embed_dim, bias=False)
        self.query_embed = nn.Parameter(class_embed)
        self.transformer = nn.Transformer(d_model=self.embed_dim, batch_first=True)
        self.group_linear = GroupWiseLinear(self.num_classes, self.embed_dim, bias=True)

    def forward(self, images):
        b, t, c, h, w = images.size()
        x = self.backbone(images)[0]
        x = self.linear1(F.adaptive_avg_pool1d(x.transpose(1, 2), t).transpose(1, 2))
        x = self.pos_encod(x)
        video_features = self.image_model(images.reshape(b*t, c, h, w))[1].reshape(b, t, -1).mean(dim=1, keepdim=True)
        query_embed = self.linear2(torch.concat((self.query_embed.unsqueeze(0).repeat(b, 1, 1), video_features.repeat(1, self.num_classes, 1)), 2))
        hs = self.transformer(x, query_embed) # b, t, d
        out = self.group_linear(hs)
        return out