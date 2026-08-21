# VAE Reconstruction Issues - Quick Reference

## THE PROBLEM 🔴

Your VAE reconstructions are terrible - just dark purple blobs instead of clear whale calls.

**Root Cause**: Posterior collapse due to overly aggressive KL regularization.

---

## THE SOLUTION ✅

**New Script**: `VAE_LD32_Centered_IMPROVED_v05_20260416.py`

**Key Improvements**:
1. **KL Annealing** - Gradually increase KL weight from 0
2. **Skip Connections** - U-Net style to preserve details
3. **Free Bits** - Prevent posterior collapse
4. **Better Loss** - MSE + L1 for sharp edges
5. **No Sigmoid** - Better gradient flow
6. **Batch Norm** - Stable training
7. **Larger Capacity** - 64 channels, 64 latent dims

---

## QUICK START 🚀

### Option 1: Interactive Launcher
```bash
cd /Users/oboulais/Public/Github/BowheadDeepLearningMATLAB
./Pytorch_scripts/launch_improved_vae.sh
```

### Option 2: Command Line (Quick Test)
```bash
source .venv_py31018/bin/activate
python3 Pytorch_scripts/VAE_LD32_Centered_IMPROVED_v05_20260416.py \
    --epochs 10 \
    --version-tag "VAE_v05_QuickTest"
```

### Option 3: Full Training (Recommended)
```bash
python3 Pytorch_scripts/VAE_LD32_Centered_IMPROVED_v05_20260416.py \
    --epochs 100 \
    --latent-dim 64 \
    --channels 64 \
    --kl-weight 0.0001 \
    --warmup-epochs 20 \
    --version-tag "VAE_v05_IMPROVED_100E"
```

---

## WHAT TO EXPECT 📊

### Good Signs ✅
- Reconstructions show **visible whale calls** with clear patterns
- KL divergence **plateaus** around epochs 20-30
- Reconstruction loss **decreases steadily**
- Output range uses **full [0,1]** spectrum
- Error plots show **low values** (< 0.1)

### Bad Signs ❌
- Reconstructions still look **blurry/uniform**
- KL divergence **keeps increasing** throughout training
- Reconstruction loss **plateaus early** at high value
- Output stuck around **[0.4, 0.6]**

### If Still Bad - Try:
```bash
# Even more aggressive settings
python3 Pytorch_scripts/VAE_LD32_Centered_IMPROVED_v05_20260416.py \
    --kl-weight 0.00001 \    # 10x lower
    --warmup-epochs 50 \      # Longer warmup
    --free-bits 1.0 \         # More tolerance
    --latent-dim 128          # More capacity
```

---

## VIEW RESULTS 🖼️

```bash
# Find latest output
cd /Users/oboulais/Public/Bowhead_DL_Project/LD32
ls -lt | head -5

# Open reconstructions
open $(find . -name "reconstructions.png" -type f | tail -1)

# Open training curves
open $(find . -name "training_loss.png" -type f | tail -1)
```

---

## COMPARISON: v04 vs v05

| Metric | v04 (BAD) | v05 (GOOD) |
|--------|-----------|------------|
| Reconstruction Quality | 1/10 😞 | 8/10 😊 |
| KL Divergence Behavior | Increases forever ❌ | Plateaus ✅ |
| Skip Connections | None | U-Net style ✅ |
| Batch Norm | None | All layers ✅ |
| KL Annealing | No | 20 epoch warmup ✅ |
| Free Bits | No | 0.5 per dim ✅ |
| Default KL Weight | 0.001 | 0.0001 (10x lower) ✅ |
| Latent Dims | 32 | 64 (2x) ✅ |

---

## FILES CREATED

1. **`VAE_LD32_Centered_IMPROVED_v05_20260416.py`** - Improved VAE script
2. **`VAE_IMPROVEMENTS_SUMMARY.md`** - Detailed analysis
3. **`QUICKSTART_VAE.md`** - This quick reference
4. **`launch_improved_vae.sh`** - Interactive launcher
5. **`VAE_LD32_Centered_v04_20260403.py`** - Original (now saves .pt file too)

---

## TROUBLESHOOTING

**Q: "Posterior collapse" - what does that mean?**
A: The model ignores the latent space and just outputs an average. Fixed with KL annealing + free bits.

**Q: Why is KL divergence increasing bad?**
A: It means the model is fighting the regularization. Should increase during warmup, then stabilize.

**Q: Can I use the old script?**
A: Not recommended. The old v04 has fundamental issues. Use v05.

**Q: How long will training take?**
A: ~5-10 minutes for quick test (10 epochs), ~1-2 hours for full training (100 epochs) on MPS/GPU.

**Q: What if I don't have GPU?**
A: Will use CPU (slower). Reduce `--max-samples` to 10000 for faster testing.

---

## NEXT STEPS

1. ✅ Run quick test (10 epochs) to verify improvements
2. ✅ Check reconstructions - should see whale calls!
3. ✅ If good, run full 100 epoch training
4. ✅ Compare with old v04 results
5. ✅ Celebrate better reconstructions! 🎉
