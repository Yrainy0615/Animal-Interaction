# 学習が進まない問題の分析レポート

## 実行日時
2025-10-20

## 問題の概要
`torchrun --nproc_per_node=1 --master_port=12345 main.py --config configs/mmnet/XCLIP-16-8.yaml` を実行しても、**学習が全く進まない**問題が発生しています。

## 主な症状

### 1. 損失値が完全に固定されている
- **観測された損失値**: `tot_loss 0.3105` (全エポック、全ステップで同一)
- **エポック数**: 30エポック実行
- **ステップ数**: 各エポック約1,819ステップ
- **変化**: 損失値が一切変動しない

```
[2025-10-19 10:38:20] Train: [0/30][0/1819]    tot_loss 0.3105 (0.3105)
[2025-10-19 10:39:18] Train: [0/30][50/1819]   tot_loss 0.3105 (0.3105)
...
[2025-10-20 06:56:01] Train: [29/30][1800/1819] tot_loss 0.3105 (0.3105)
```

### 2. 検証精度が極めて低い
- **Acc@1**: 0.0742 (7.42%) - 12クラス分類でランダム推測(8.33%)に近い
- **Acc@5**: 0.6003 (60.03%)
- **mAP**: 0.0833
- **全エポックで精度が一切向上していない**

## 根本原因の分析

### 原因1: モデルが完全にランダムな予測を出力している

損失値 `0.3105` の意味:
```python
# 12クラス分類での理論的な一様分布損失
expected_loss = -log(1/12) = 2.4849

# ACCUMULATION_STEPS = 8 で除算
displayed_loss = 2.4849 / 8 = 0.3106 ≈ 0.3105
```

**結論**: モデルは毎回すべてのクラスに対して等確率(1/12)の予測を出力しており、学習が一切進んでいません。

### 原因2: 勾配が適用されていない可能性

以下の点から、勾配が正しく適用されていない可能性が高いです:

#### A. モデルが `eval()` モードで返される
`models/model_builder.py:401`:
```python
def build_model(state_dict: dict, ...):
    ...
    return model.eval()  # ← 問題箇所
```

**影響**:
- BatchNormalization層が学習モードにならない
- Dropout層が無効化される
- `model.training = False` の状態で返される

ただし、`train.py:60` で `model.train()` が呼ばれているため、これだけが原因ではない可能性があります。

#### B. テキストエンコーダーの固定
`utils/optimizer.py` の `fix_text()` 関数:
```python
def fix_text(model):
    for name, param in model.named_parameters():
        if "visual." in name or "mit" in name or "prompts" in name or 'graph' in name:
            continue
        else:
            param.requires_grad=False
```

設定: `FIX_TEXT: True`

**影響**: テキストエンコーダーのパラメータは固定される(これは正常)

#### C. 勾配累積の実装
`train.py:138-170`:
```python
total_loss = criterion(output, label_id)
total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS  # ← 損失を8で除算

if config.TRAIN.ACCUMULATION_STEPS == 1:
    optimizer.zero_grad()

if use_amp:
    scaler.scale(total_loss).backward()
else:
    total_loss.backward()
    
if config.TRAIN.ACCUMULATION_STEPS > 1:
    if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        ...
```

**潜在的な問題**:
1. `ACCUMULATION_STEPS = 8` の場合、8ステップごとにしか optimizer.step() が呼ばれない
2. しかし、**最初のステップでは optimizer.zero_grad() が呼ばれていない**
3. 勾配が正しく累積・適用されていない可能性

### 原因3: ラベルの形状とインデックスの問題

`train.py:110-136` でラベルの形状変換を行っていますが、以下の懸念があります:

```python
if label_id.ndim == 2:
    if label_id.shape[1] > 1:
        label_id = label_id.argmax(dim=1)  # One-hot → クラスインデックス
    else:
        label_id = label_id.squeeze(dim=1)

label_id = label_id.squeeze().long()

# 1-based → 0-based 変換
if label_id.max() >= config.DATA.NUM_CLASSES:
    logger.warning(f"Label index >= {config.DATA.NUM_CLASSES} detected.")
    label_id = label_id - 1
```

**問題の可能性**:
- ラベルが常に同じ値になっている可能性
- ラベルの変換ロジックが正しく機能していない可能性

## 検証すべき項目

### 優先度: 高

1. **勾配の流れを確認**
   ```python
   # train.py の学習ループ内に追加
   for name, param in model.named_parameters():
       if param.requires_grad and param.grad is not None:
           print(f'{name}: grad_norm={param.grad.norm().item():.6f}')
   ```

2. **optimizer.step() の呼び出し回数を確認**
   ```python
   step_count = 0
   if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
       step_count += 1
       print(f'Optimizer step called: {step_count}')
   ```

3. **モデルの出力を確認**
   ```python
   print(f'Output shape: {output.shape}')
   print(f'Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}')
   print(f'Output softmax: {output.softmax(dim=-1)[0]}')
   ```

4. **ラベルの値を確認**
   ```python
   print(f'Label shape: {label_id.shape}')
   print(f'Label values: {label_id}')
   print(f'Label range: min={label_id.min()}, max={label_id.max()}')
   ```

### 優先度: 中

5. **学習可能なパラメータ数を確認**
   - ログに出力されている "trainable paras number" の値を確認
   - 0やあまりにも少ない場合は、パラメータが正しく学習可能になっていない

6. **BatchNorm/Dropout の動作を確認**
   ```python
   print(f'Model training mode: {model.training}')
   for module in model.modules():
       if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
           print(f'{module.__class__.__name__}: training={module.training}')
   ```

## 推奨される修正

### 修正1: build_model() の戻り値を変更

`models/model_builder.py:401`:
```python
# 修正前
return model.eval()

# 修正後
return model  # eval() を削除
```

### 修正2: 勾配累積のロジックを修正

`train.py:141-170`:
```python
# 修正前
if config.TRAIN.ACCUMULATION_STEPS == 1:
    optimizer.zero_grad()

# 修正後
if (idx == 0) or (config.TRAIN.ACCUMULATION_STEPS == 1) or ((idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0):
    optimizer.zero_grad()
```

または、より明確に:
```python
# 各ステップの最初に zero_grad を呼ぶべきか判断
should_zero_grad = (
    (idx % config.TRAIN.ACCUMULATION_STEPS == 0) or 
    (config.TRAIN.ACCUMULATION_STEPS == 1)
)

if should_zero_grad:
    optimizer.zero_grad()

# Forward pass
with autocast(enabled=use_amp):
    ...
    total_loss = criterion(output, label_id)
    total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

# Backward pass
if use_amp:
    scaler.scale(total_loss).backward()
else:
    total_loss.backward()

# Optimizer step (accumulation完了時のみ)
if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    lr_scheduler.step_update(epoch * num_steps + idx)
```

### 修正3: デバッグ出力を追加

`train.py` の学習ループに以下を追加:
```python
if idx % 50 == 0:  # 50ステップごとに出力
    # 勾配の確認
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    if grad_norms:
        print(f'Step {idx}: grad_norm avg={np.mean(grad_norms):.6f}, max={np.max(grad_norms):.6f}')
    else:
        print(f'Step {idx}: No gradients found!')
    
    # 出力の確認
    with torch.no_grad():
        output_probs = output.softmax(dim=-1)
        print(f'Output entropy: {-(output_probs * torch.log(output_probs + 1e-8)).sum(dim=-1).mean():.4f}')
        print(f'Expected entropy (uniform): {torch.log(torch.tensor(float(config.DATA.NUM_CLASSES))):.4f}')
```

## 即座に実行可能なデバッグコマンド

以下のコマンドで、デバッグ情報を含む短時間の学習を実行できます:

```bash
# デバッグ用に1エポックのみ実行
torchrun --nproc_per_node=1 --master_port=12345 main.py \
  --config configs/mmnet/XCLIP-16-8.yaml \
  --opts TRAIN.EPOCHS 1 PRINT_FREQ 10
```

## まとめ

**最も可能性の高い原因**:
1. `model.eval()` で返されたモデルが、`model.train()` で正しく訓練モードに切り替わっていない可能性
2. 勾配累積のロジックに問題があり、optimizer.step() が正しく呼ばれていない
3. モデルのパラメータがすべて固定されている、または勾配が計算されていない

**次のステップ**:
1. 上記の「検証すべき項目」セクションのデバッグコードを追加
2. 「推奨される修正」セクションの修正を適用
3. 短時間の学習を実行して、勾配が正しく流れているか確認
4. 問題が解決しない場合は、さらに詳細なログ出力を追加

## 付録: 設定値の確認

### 現在の設定 (configs/mmnet/XCLIP-16-8.yaml)
```yaml
TRAIN:
  BATCH_SIZE: 8
  ACCUMULATION_STEPS: 8  # 実効バッチサイズ = 8 * 8 = 64
  EPOCHS: 30
  LR: 8e-06
  OPTIMIZER: adamw
  OPT_LEVEL: O1  # Mixed precision training
  
MODEL:
  FIX_TEXT: True  # テキストエンコーダーを固定

DATA:
  NUM_CLASSES: 12
  NUM_ANIMAL_CLASSES: 173
```

### ログから確認できる情報
- 学習率: 非常に小さい値から開始 (0.000000000 → 徐々に増加)
- これは warmup の影響
- WARMUP_EPOCHS: 5.0 なので、最初の5エポックは学習率が低い
- しかし、warmup 終了後も損失が変化していない

**結論**: 学習率の問題ではなく、根本的に勾配が適用されていない可能性が高いです。
