import torch
import sys

# Path to the original model

# LD16
original_model_path = "/Users/oboulais/Desktop/Bowhead_DL_Project/LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/trained_model/autoencoder_clean.pth"
# original_model_path = "/Users/oboulais/Desktop/Bowhead_DL_Project/LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir/trained_model/autoencoder_clean.pt"

# original_model_path = "/Users/oboulais/Desktop/Bowhead_DL_Project/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth"
# original_model_path = "/Users/oboulais/Desktop/Bowhead_DL_Project/LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir/trained_model/autoencoder_clean.pt"

def load_checkpoint(path):
    chk = torch.load(path, map_location='cpu')
    # If checkpoint wraps the state_dict inside another dict, extract it
    if isinstance(chk, dict):
        if 'state_dict' in chk:
            return chk['state_dict']
        if 'model_state_dict' in chk:
            return chk['model_state_dict']
        # If it's a flat OrderedDict of params, return it
        # else assume it's the state_dict
    return chk

state_dict = load_checkpoint(original_model_path)

print("\n--- checkpoint keys and tensor shapes ---\n")
if isinstance(state_dict, dict):
    for k, v in state_dict.items():
        try:
            print(f"{k}: {tuple(v.shape)}")
        except Exception:
            print(f"{k}: {type(v)}")
else:
    print(type(state_dict))

print("\n--- end ---\n")

print("Inspection complete. Continuing to reconstruct and trace the model...")

# ---- If you want to automatically reconstruct and trace the model, uncomment below ----
# The following block instantiates the training architecture used in this repo
# (ImprovedAutoencoder), loads the saved state_dict, traces it with a dummy
# input sized for (1,1,121,104), and writes a traced .pt file compatible with
# MATLAB converter.

if __name__ == '__main__':
    try:
        # Import model definition from repository
        import os
        repo_scripts = os.path.expanduser("/Users/oboulais/Desktop/Github/BowheadDeepLearningMATLAB/Pytorch_scripts")
        if repo_scripts not in sys.path:
            sys.path.insert(0, repo_scripts)
        from Apply_Autoencoder import ImprovedAutoencoder

        # Instantiate model with parameters inferred from checkpoint
        model = ImprovedAutoencoder(nrow=121, ncol=104, latent_dim=16, base_channels=32, extra_conv=False)
        model.load_state_dict(state_dict)
        model.eval()

        # Create dummy input: batch 1, 1 channel, 121x104
        dummy_input = torch.randn(1, 1, 121, 104)

        traced_model = torch.jit.trace(model, dummy_input)
        traced_path = original_model_path.replace('.pt', '_traced.pt')
        torch.jit.save(traced_model, traced_path)
        print(f"Traced model saved to {traced_path}")
    except Exception as e:
        print("Failed to auto-trace model:", e)