# Quick Start Guide - Running Animal-CLIP with torch.cuda.amp

## What Was Fixed

The codebase has been migrated from `apex.amp` to `torch.cuda.amp`, resolving version compatibility issues while maintaining mixed precision training functionality.

## Prerequisites

- Python 3.6+
- PyTorch 1.6+ (tested with 2.4.1+cu121)
- CUDA-capable GPU

## Running Your Command

You can now run your original command without apex dependency issues:

```bash
torchrun --nproc_per_node=1 --master_port=12345 main.py \
   --config configs/mmnet/XCLIP-16-8.yaml \
   --only_test \
   --opts MODEL.RESUME /mnt/nfs/mammal_net/models/Animal-CLIP/mmnet_best.pth
```

## What Changed

### Import Statements
- ❌ **Old**: `from apex import amp`
- ✅ **New**: `from torch.cuda.amp import autocast, GradScaler`

### Mixed Precision Training
The code now uses PyTorch's native automatic mixed precision:
- **Forward pass**: Wrapped with `autocast()` context manager
- **Backward pass**: Uses `GradScaler` to scale gradients
- **Optimizer step**: Uses `scaler.step()` and `scaler.update()`

### Configuration
Your existing configuration files work as-is:
- `OPT_LEVEL='O0'`: Full FP32 precision (no AMP)
- `OPT_LEVEL='O1'` or `'O2'`: Mixed precision enabled

## Files Modified

1. **main.py** - Updated initialization and training call
2. **train.py** - Refactored training loop for torch.cuda.amp
3. **val.py** - Updated inference to use autocast

## Verification

To verify everything works:

```bash
# Test PyTorch and CUDA
python3 -c "import torch; from torch.cuda.amp import autocast, GradScaler; print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# This should output:
# PyTorch 2.4.1+cu121 with CUDA 12.1
```

## Troubleshooting

### If you see "ModuleNotFoundError: No module named 'apex'"
✅ This is now fixed! The code no longer depends on apex.

### If you see other missing module errors
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### If mixed precision is not working
Check that:
1. Your GPU supports mixed precision (Compute Capability 7.0+)
2. `config.TRAIN.OPT_LEVEL` is set to something other than 'O0'
3. CUDA is available: `python3 -c "import torch; print(torch.cuda.is_available())"`

## Additional Information

For detailed information about the migration, see `APEX_TO_TORCH_AMP_MIGRATION.md`.

## Support

The changes maintain backward compatibility with your existing configs and should work identically to the apex version, but without the dependency issues.
