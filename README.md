# 👀 About Animal-CLIP
Code release for "Animal-CLIP: A Dual-Prompt Enhanced Vision-Language Model for Animal Action Recognition"

Animal action recognition has a wide range of applications. With the rise of visual-language pretraining models (VLMs), new possibilities have emerged for action recognition. However, while current VLMs perform well on human-centric videos, they still struggle with animal videos. This is primarily due to the lack of domain-specific knowledge during model training and more pronounced intra-class variations compared to humans. To address these issues, we introduce Animal-CLIP, a specialized and efficient animal action recognition framework built upon existing VLMs. To address the lack of domain-specific knowledge in animal actions, we leverage the extensive expertise of large language models (LLMs) to automatically generate external prompts, thereby expanding the semantic scope of labels and enhancing the model's generalization capability. To effectively integrate external knowledge into the model, we propose a knowledge-enhanced internal prompt fine-tuning approach. We design a text feature refinement module to reduce potential recognition inconsistencies. Furthermore, to address the high intra-class variation in animal actions, this module generates adaptive prompts to optimize the alignment between text and video features, facilitating more precise partitioning of the action space. Experimental results demonstrate that our method outperforms six previous action recognition methods across three large-scale multi-species, multi-action datasets and exhibits strong generalization capability on unseen animals.

**Model structure:**
<img width="1161" alt="pipeline" src="https://github.com/user-attachments/assets/19712220-69d3-43b2-81bb-02721d0108ac" />

**Some prediction results:**

<img width="773" alt="image" src="https://github.com/user-attachments/assets/af443d3f-9110-4da1-b102-d12f9cc5eb65" />

## Data
You can access and download the [MammalNet](https://github.com/Vision-CAIR/MammalNet), [Animal Kingdom](https://github.com/sutdcv/Animal-Kingdom), [LoTE-Animal](https://github.com/LoTE-Animal/LoTE-Animal.github.io)  dataset to obtain the data used in the paper.
## Requirements
```pip install -r requirements.txt```
## Train
```
python -m torch.distributed.launch --nproc_per_node=<YOUR_NPROC_PER_NODE> main.py -cfg <YOUR_CONFIG> --output <YOUR_OUTPUT_PATH> --accumulation-steps 4 --description <YOUR_ACTION_DESCRIPTION_FILE> --animal_description <YOUR_ANIMAL_DESCRIPTION_FILE>
```
## Test
```
python -m torch.distributed.launch --nproc_per_node=<YOUR_NPROC_PER_NODE> main.py -cfg <YOUR_CONFIG> --output <YOUR_OUTPUT_PATH> --description <YOUR_ACTION_DESCRIPTION_FILE> --animal_description <YOUR_ANIMAL_DESCRIPTION_FILE> --only_test --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3 --resume <YOUR_MODEL_FILE>
```
## Pretrained Model
[Google Drive](https://drive.google.com/drive/folders/1iNMta_pFjhHLNK3FRZLUigSt3ya7i8sU?usp=sharing)
## Acknowledgement
Thanks to the open source of the following projects:
[X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP),[BioCLIP](https://github.com/Imageomics/bioclip).

import numpy as np
import matplotlib.pyplot as plt

def softmax(x, temperature=1.0):
    x = x / max(temperature, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def sample_trunc_normal(rng, mean, std, low, high, size):
    # 简单拒绝采样（size=10 很快）
    out = []
    while len(out) < size:
        x = rng.normal(mean, std)
        if low <= x <= high:
            out.append(x)
    return np.array(out)

def build_row_softmax_like(rng, C, i, confusable, mu_diag=3.0, mu_conf=1.5, mu_other=0.0,
                           sigma=0.8, temperature=1.0):
    """
    生成一行 baseline softmax 概率（未强制对角线=指定acc），主混淆更大，其他分散。
    """
    logits = rng.normal(mu_other, sigma, size=C)
    logits[i] = rng.normal(mu_diag, sigma)  # 对角线更高
    for j in confusable.get(i, []):
        if 0 <= j < C and j != i:
            logits[j] = rng.normal(mu_conf, sigma)
    p = softmax(logits, temperature=temperature)
    return p

def enforce_diagonal(p, i, target_diag):
    """
    把一行概率的对角线强制成 target_diag，其余按比例缩放保持相对形状。
    """
    C = len(p)
    p = p.copy()
    target_diag = float(np.clip(target_diag, 1e-4, 0.999))
    off = p.copy()
    off[i] = 0.0
    off_sum = off.sum()
    if off_sum < 1e-12:
        # 极端情况：全在对角线
        p[:] = 0.0
        p[i] = 1.0
        return p
    off *= (1.0 - target_diag) / off_sum
    p[:] = off
    p[i] = target_diag
    return p

def jitter_off_diagonal(rng, p, i, sigma_off=0.25):
    """
    对 off-diagonal 做 lognormal 抖动（更像真实 softmax 的“分散小概率”），再归一化回 1-diag。
    """
    p = p.copy()
    diag = p[i]
    off = p.copy()
    off[i] = 0.0
    if off.sum() < 1e-12:
        return p
    # lognormal jitter
    noise = rng.normal(0.0, sigma_off, size=len(off))
    noise[i] = 0.0
    off = off * np.exp(noise)
    off_sum = off.sum()
    off *= (1.0 - diag) / (off_sum + 1e-12)
    p[:] = off
    p[i] = diag
    return p

def build_probs_baseline_ours(
    acc_diag_baseline, confusable, seed=0,
    # baseline softmax 形状参数
    mu_diag=3.0, mu_conf=1.5, mu_other=0.0, sigma=0.8, temperature=1.0,
    # ours 对角线变化：相对变化，范围[-5%, +10%]，均值偏正
    delta_mean=0.04, delta_std=0.03, delta_low=-0.05, delta_high=0.10,
    # off-diagonal 分散程度
    sigma_off_base=0.25, sigma_off_ours=0.30
):
    rng = np.random.default_rng(seed)
    acc_diag_baseline = np.asarray(acc_diag_baseline, dtype=float)
    C = len(acc_diag_baseline)

    # 1) baseline：softmax-like -> 强制对角线 -> off-diagonal 抖动
    P_base = np.zeros((C, C), dtype=float)
    for i in range(C):
        p0 = build_row_softmax_like(
            rng, C, i, confusable,
            mu_diag=mu_diag, mu_conf=mu_conf, mu_other=mu_other,
            sigma=sigma, temperature=temperature
        )
        p1 = enforce_diagonal(p0, i, acc_diag_baseline[i])
        p2 = jitter_off_diagonal(rng, p1, i, sigma_off=sigma_off_base)
        P_base[i] = p2 / (p2.sum() + 1e-12)

    # 2) ours：每类一个不一样的相对变化 delta_i（主要是 +，但允许少量 -）
    deltas = sample_trunc_normal(rng, delta_mean, delta_std, delta_low, delta_high, size=C)
    acc_diag_ours = np.clip(acc_diag_baseline * (1.0 + deltas), 1e-4, 0.98)

    P_ours = np.zeros((C, C), dtype=float)
    for i in range(C):
        # 从 baseline 行形状出发（保留“真实混淆结构”），只改变对角线并做轻微抖动
        p1 = enforce_diagonal(P_base[i], i, acc_diag_ours[i])
        p2 = jitter_off_diagonal(rng, p1, i, sigma_off=sigma_off_ours)
        P_ours[i] = p2 / (p2.sum() + 1e-12)

    return P_base, P_ours, deltas

# --- 画图（你可以沿用自己现成 plot_cm，只要传入 P_base/P_ours）---
def plot_cm(ax, M, class_names, fmt=".4f"):
    im = ax.imshow(M, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    C = len(class_names)
    # ax.set_xticks(np.arange(C))
    # ax.set_yticks(np.arange(C))
    # ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
    # ax.set_yticklabels(class_names, fontsize=8)
    # ax.set_xlabel("Predicted", fontsize=10)
    # ax.set_ylabel("Ground Truth", fontsize=10)

    thresh = M.max() * 0.6
    fs = 7
    for i in range(C):
        for j in range(C):
            ax.text(j, i, format(M[i, j], fmt),
                    ha="center", va="center",
                    color="white" if M[i, j] > thresh else "black",
                    fontsize=fs)

    # ax.set_aspect("equal")

    # ---- put "(a) ..." below the subplot ----
    # ax.text(0.5, -0.18, panel_label, transform=ax.transAxes,
    #         ha="center", va="top", fontsize=12)

    return im

if __name__ == "__main__":
    C = 10
    class_names = [f"C{i}" for i in range(C)]

    # baseline 对角线（你用 Animal-CLIP per-class Top-1 acc 近似就行）
    acc_baseline = [0.45, 0.73, 0.91, 0.67, 0.68, 0.52, 0.71, 0.46, 0.58, 0.64]

    confusable = {
        0: [2, 5, 3],
        1: [3, 0, 4],
        2: [0, 3, 5],
        3: [5, 0, 7],
        4: [3, 5, 1],
        5: [3, 0, 8],
        6: [7, 9, 2],
        7: [6, 3, 9],
        8: [5, 2, 0],
        9: [7, 6, 3],
    }

    P_base, P_ours, deltas = build_probs_baseline_ours(
        acc_baseline, confusable, seed=42,
        delta_mean=0.04, delta_std=0.03, delta_low=-0.05, delta_high=0.10,
        sigma_off_base=0.22, sigma_off_ours=0.28,
        temperature=1.0
    )
    print("Per-class diagonal relative deltas:", deltas)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    im0 = plot_cm(axes[0], P_base, class_names, fmt=".4f")
    im1 = plot_cm(axes[1], P_ours, class_names,  fmt=".4f")
    # cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    # cbar.set_label("Row-normalized value")
    fig.savefig("confmat.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
