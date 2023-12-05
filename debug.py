import torch


model_path = "save/humanml/model000500000.pt"
opt_path = "save/humanml/opt000500000.pt"


model = torch.load(model_path, map_location="cpu")

opt = torch.load(opt_path, map_location="cpu")


print("a")