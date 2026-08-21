# Virtual Environment Setup Guide

## Quick Start

### 1. Create Virtual Environment (One-time setup)
```bash
cd /Users/oboulais/Public/Bowhead_DL_Project
chmod +x setup_venv.sh activate_venv.sh
./setup_venv.sh
```

This will:
- Create a virtual environment named `venv_bowhead`
- Install all required dependencies:
  - PyTorch (≥2.0.0)
  - NumPy (≥1.24.0)
  - Matplotlib (≥3.7.0)
  - SciPy (≥1.10.0)
  - scikit-learn (≥1.3.0)
  - UMAP (optional, ≥0.5.3)
  - Jupyter & IPython (optional)

### 2. Activate Environment (Every time you work)
```bash
source activate_venv.sh
```

Or manually:
```bash
source venv_bowhead/bin/activate
```

### 3. Deactivate Environment
```bash
deactivate
```

## Running Scripts

Once the environment is activated, run your scripts normally:

```bash
# Activate environment
source activate_venv.sh

# Run multi-feature autoencoder
python3 Autoencoder_MultiFeature_v01.py --mode snr_ntv --database auto --epochs 10

# Run original autoencoder
cd LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir
python3 Autoencoder_v02_LD32_20251118.py

# Deactivate when done
deactivate
```

## Updating Dependencies

To add new packages:
1. Add them to `requirements.txt`
2. Reinstall:
   ```bash
   source activate_venv.sh
   pip install -r requirements.txt
   ```

## Verifying Installation

Check installed packages:
```bash
source activate_venv.sh
pip list
```

Test imports:
```bash
python3 -c "
import torch
import numpy as np
import matplotlib
import scipy
from sklearn.manifold import TSNE
import umap
print('✓ All imports successful!')
"
```

## Apple Silicon (M1/M2/M3) Notes

PyTorch will automatically use Apple Metal (MPS) for GPU acceleration if available. Check with:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Troubleshooting

**Issue**: `setup_venv.sh: Permission denied`  
**Fix**: `chmod +x setup_venv.sh`

**Issue**: Import errors after activation  
**Fix**: Delete and recreate the venv:
```bash
rm -rf venv_bowhead
./setup_venv.sh
```

**Issue**: UMAP installation fails  
**Fix**: UMAP is optional. The scripts will work without it. To install separately:
```bash
source activate_venv.sh
pip install umap-learn
```

## Environment Details

- **Location**: `/Users/oboulais/Public/Bowhead_DL_Project/venv_bowhead`
- **Python Version**: Uses system Python 3.x
- **Isolation**: Completely isolated from system packages
- **Portability**: Not portable (paths are absolute)

## Best Practices

1. **Always activate before working**:
   ```bash
   source activate_venv.sh
   ```

2. **Never commit `venv_bowhead/`** (already in .gitignore)

3. **Update `requirements.txt`** when adding new dependencies

4. **Recreate periodically** to keep dependencies fresh:
   ```bash
   ./setup_venv.sh
   ```
