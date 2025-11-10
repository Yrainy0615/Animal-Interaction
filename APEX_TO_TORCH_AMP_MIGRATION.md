# Migration from apex.amp to torch.cuda.amp

This document describes the changes made to replace `apex.amp` with native PyTorch `torch.cuda.amp` for automatic mixed precision training.

## Summary

The codebase has been successfully migrated from NVIDIA Apex AMP to PyTorch's native automatic mixed precision (AMP) implementation. This resolves version compatibility issues with Apex while maintaining mixed precision training functionality.

## Changes Made

### 1. **main.py**
- **Line 21-22**: Replaced `from apex import amp` with `from torch.cuda.amp import autocast, GradScaler`
- **Line 150-153**: Replaced `amp.initialize()` with `GradScaler()` initialization
  - Before: `model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)`
  - After: Created a `scaler` object for mixed precision training
- **Line 188**: Updated function call to pass `scaler` to `train_one_epoch()`

### 2. **train.py**
- **Line 21-22**: Replaced `from apex import amp` with `from torch.cuda.amp import autocast, GradScaler`
- **Line 56**: Updated function signature to accept `scaler` parameter
- **Line 93-120**: Refactored training loop to use `autocast` and `GradScaler`:
  - Wrapped forward pass with `autocast(enabled=use_amp)`
  - Replaced `amp.scale_loss()` with `scaler.scale(total_loss).backward()`
  - Updated optimizer step to use `scaler.step()` and `scaler.update()`

### 3. **val.py**
- **Line 21-22**: Replaced `from apex import amp` with `from torch.cuda.amp import autocast`
- **Line 86-114**: Wrapped inference forward pass with `autocast(enabled=use_amp)`
  - Removed manual `.half()` conversion for O2 opt level
  - Now using automatic mixed precision during inference

## Key Differences Between apex.amp and torch.cuda.amp

| Aspect | apex.amp | torch.cuda.amp |
|--------|----------|----------------|
| **Initialization** | `amp.initialize(model, optimizer, opt_level='O2')` | `scaler = GradScaler()` |
| **Forward Pass** | Automatic after initialization | Wrapped with `autocast()` context |
| **Backward Pass** | `with amp.scale_loss(loss, optimizer) as scaled_loss: scaled_loss.backward()` | `scaler.scale(loss).backward()` |
| **Optimizer Step** | `optimizer.step()` | `scaler.step(optimizer); scaler.update()` |
| **Inference** | Manual `.half()` for O2 | `with autocast():` context |

## Compatibility

- **PyTorch Version**: Requires PyTorch 1.6 or later (tested with 2.4.1+cu121)
- **No external dependencies**: Removes dependency on NVIDIA Apex
- **Behavior**: Equivalent to Apex O1/O2 mixed precision training

## Testing

To verify the changes work correctly, run your training command:

```bash
torchrun --nproc_per_node=1 --master_port=12345 main.py \
   --config configs/mmnet/XCLIP-16-8.yaml \
   --only_test \
   --opts MODEL.RESUME /mnt/nfs/mammal_net/models/Animal-CLIP/mmnet_best.pth
```

## Notes

- The `OPT_LEVEL` configuration parameter is still respected:
  - `O0`: Full FP32 training (no mixed precision)
  - Other values (`O1`, `O2`, etc.): Mixed precision training enabled
- Performance should be equivalent to Apex AMP
- Memory usage should be similar to Apex O1/O2 optimization levels
- The code automatically handles gradient accumulation with mixed precision

## Rollback

If you need to rollback to apex.amp, simply:
1. Uncomment the `# from apex import amp` lines
2. Comment out the `from torch.cuda.amp import ...` lines
3. Revert the code changes in the training and validation loops

However, this is not recommended due to the version compatibility issues with Apex.
