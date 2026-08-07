# import torch
# import sys
# import pickle
# import argparse
# sys.path.append("/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/code")
# from network import UNet_flex



# # model_dir = "/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/models_trained/UNet_full_240541imgs_1000epochs"
# model_dir = "/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/Denoiser_Reconstruction/assets"

# with open(f"{model_dir}/exp_arguments.pkl", "rb") as f:
#     saved_arguments_dict = pickle.load(f)
# args = argparse.Namespace(**saved_arguments_dict)

# model = UNet_flex(args)

# # weights_only=False because these checkpoints were saved with an older
# # PyTorch pickle format.
# state_dict = torch.load(f"{model_dir}/model.pt", map_location="cpu", weights_only=False)

# fixed_state_dict = {}
# for key, value in state_dict.items():
#     fixed_key = key.removeprefix("module.")
#     fixed_state_dict[fixed_key] = value

# model.load_state_dict(fixed_state_dict)


# model.eval()
# print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# path = "/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/models_trained/UNet_full_240541imgs_1000epochs"

# model = UNet_flex()

# state_dict = torch.load(path, weights_only=True)
# model.load_state_dict(state_dict)
# model.eval()
# print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


import torch
import sys
import argparse
sys.path.append("/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/Denoiser_Reconstruction")
from models.denoiser import Denoiser


model_dir = "/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/Denoiser_Reconstruction/assets"

# conv3_ln.pt has no exp_arguments.pkl; these are the parse_args() defaults
# from Denoiser_Reconstruction/utils/helper.py for the network architecture.
args = argparse.Namespace(
    padding=1,
    kernel_size=3,
    num_kernels=64,
    num_layers=20,
    im_channels=3,
)

model = Denoiser(args)

state_dict = torch.load(f"{model_dir}/conv3_ln.pt", map_location="cpu", weights_only=True)

fixed_state_dict = {}
for key, value in state_dict.items():
    fixed_key = key.removeprefix("module.")
    fixed_state_dict[fixed_key] = value

model.load_state_dict(fixed_state_dict)


model.eval()
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
