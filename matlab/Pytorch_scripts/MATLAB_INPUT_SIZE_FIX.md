# MATLAB Import Guide - Fix Input Size Warning

## The Problem

When importing PyTorch models into MATLAB without specifying `InputSize`, you get this warning:
```
The software detected a 2-D image input and automatically added an 
imageInputLayer. Edit the InputSize property of the imageinput_1 layer 
before you use the network.
```

## The Solution

**Always specify `InputSize` when importing:**

```matlab
% ❌ WRONG - Causes warning
net = importNetworkFromPyTorch('autoencoder_clean_matlab_compatible.pt');

% ✅ CORRECT - No warning
net = importNetworkFromPyTorch('autoencoder_clean_matlab_compatible.pt', ...
    'InputSize', [121 104 1]);
```

## Complete Import Code

### Single Model Import

```matlab
% Navigate to model directory
cd('LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir/trained_model')

% Import with explicit InputSize
net = importNetworkFromPyTorch('autoencoder_clean_matlab_compatible.pt', ...
    'InputSize', [121 104 1]);  % Height × Width × Channels

% Verify input layer (should show no warnings)
net.Layers(1)

% Test prediction
test_input = randn(121, 104, 1, 5);  % 5 samples
predictions = predict(net, test_input);
fprintf('Success! Output size: %s\n', mat2str(size(predictions)));
```

### Batch Import All Models

Use the provided MATLAB script:

```matlab
% Run the automated import script
run('import_models_matlab.m');

% This imports all 4 models:
% - models.LD16_v13_AutoManual
% - models.LD16_v14_Manual
% - models.LD32_v13_AutoManual
% - models.LD32_v14_Manual
```

## Input Size Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Height** | 121 | SNR_gram rows |
| **Width** | 104 | SNR_gram columns |
| **Channels** | 1 | Single-channel grayscale |
| **Batch** | N | Variable (any number of samples) |

**MATLAB format:** `[H W C]` = `[121 104 1]`  
**PyTorch format (internal):** `[B C H W]` = `[N 1 121 104]`

MATLAB's `importNetworkFromPyTorch` automatically handles the dimension reordering.

## Using Imported Models

### Process Single Sample

```matlab
% Load a whale call spectrogram
data = load('BCB_Whale_Datasets/...dir/S308A0T20080828T000045_Type0.mat');
snr_gram = data.SNR_gram;  % 121×104 matrix

% Reshape for network input (add batch and channel dimensions)
input_data = reshape(snr_gram, [121, 104, 1, 1]);

% Run autoencoder
reconstruction = predict(net, input_data);

% Extract result (remove batch and channel dimensions)
reconstructed_snr = squeeze(reconstruction);  % Back to 121×104

% Visualize
figure;
subplot(1,2,1); imagesc(snr_gram); title('Original');
subplot(1,2,2); imagesc(reconstructed_snr); title('Reconstructed');
colormap jet; colorbar;

% Calculate error
error = mean(abs(snr_gram - reconstructed_snr), 'all');
fprintf('Reconstruction error: %.4f\n', error);
```

### Process Batch of Samples

```matlab
% Load multiple samples
num_samples = 10;
batch_data = zeros(121, 104, 1, num_samples);

for i = 1:num_samples
    % Load each sample
    data = load(sprintf('path/to/sample_%d.mat', i));
    batch_data(:,:,1,i) = data.SNR_gram;
end

% Batch prediction (much faster than loop)
reconstructions = predict(net, batch_data);

% Process results
for i = 1:num_samples
    original = squeeze(batch_data(:,:,1,i));
    reconstructed = squeeze(reconstructions(:,:,1,i));
    error = mean(abs(original - reconstructed), 'all');
    fprintf('Sample %d error: %.4f\n', i, error);
end
```

## Checking Import Success

### Verify No Warnings or Errors

```matlab
% Should show imageInputLayer with correct size
input_layer = net.Layers(1);
assert(strcmp(class(input_layer), 'nnet.cnn.layer.ImageInputLayer'), ...
    'First layer is not ImageInputLayer!');
assert(isequal(input_layer.InputSize, [121 104 1]), ...
    'Input size is incorrect!');

% Check for placeholder/ATEN layers (should be 0)
layer_types = string({net.Layers.Type});
placeholder_count = sum(contains(layer_types, 'Placeholder'));
aten_count = sum(contains(string({net.Layers.Name}), 'ATEN'));

assert(placeholder_count == 0, 'Found %d placeholder layers!', placeholder_count);
assert(aten_count == 0, 'Found %d ATEN layers!', aten_count);

fprintf('✓ All checks passed! Model is fully compatible.\n');
```

## Common Issues and Fixes

### Issue 1: Wrong Input Dimensions
```
Error: Invalid input data. The size of the input data does not match...
```

**Fix:** Ensure data is in MATLAB format `[H W C N]`:
```matlab
% If your data is 121×104 (2D)
input_data = reshape(snr_gram, [121, 104, 1, 1]);  % Add C and N dimensions

% If your data is already 4D, check order
size(input_data)  % Should be [121 104 1 N]
```

### Issue 2: Placeholder Warnings
```
Warning: Detected layers with custom functions...
```

**Fix:** Ensure you're using the newly generated models from `export_matlab_compatible_model_v2.py`. Delete old `_matlab_compatible.pt` files and regenerate.

### Issue 3: Dimension Mismatch Errors
```
Error: Arrays have incompatible sizes for this operation.
```

**Fix:** Check that output size matches expected:
```matlab
predictions = predict(net, input_data);
assert(isequal(size(predictions), size(input_data)), ...
    'Output size mismatch! Input: %s, Output: %s', ...
    mat2str(size(input_data)), mat2str(size(predictions)));
```

## Quick Reference

| Task | MATLAB Code |
|------|-------------|
| **Import with InputSize** | `net = importNetworkFromPyTorch('model.pt', 'InputSize', [121 104 1]);` |
| **Check input layer** | `net.Layers(1)` |
| **Prepare data** | `input = reshape(snr_gram, [121, 104, 1, 1]);` |
| **Run prediction** | `output = predict(net, input);` |
| **Extract result** | `result = squeeze(output);` |
| **Verify no ATEN** | `sum(contains(string({net.Layers.Name}), 'ATEN'))` |

## Files Created

1. **[export_matlab_compatible_model_v2.py](export_matlab_compatible_model_v2.py)** - Model export script
2. **[import_models_matlab.m](import_models_matlab.m)** - Automated batch import script ⭐
3. **autoencoder_clean_matlab_compatible.pt** - MATLAB-compatible models (4 files in LD16/LD32)

## Next Steps

1. **In MATLAB**, navigate to project directory:
   ```matlab
   cd('/Users/oboulais/Public/Bowhead_DL_Project')
   ```

2. **Run automated import:**
   ```matlab
   run('import_models_matlab.m')
   ```

3. **Use imported models:**
   ```matlab
   % Models are stored in 'models' struct
   models.LD32_v14_Manual
   ```

**No more warnings!** 🎉
