import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, NormalizeIntensityd, ConvertToMultiChannelBasedOnBratsClassesd
from monai.data import Dataset
from monai.inferers import sliding_window_inference
from functools import partial
import matplotlib.pyplot as plt
import nibabel as nib
from monai.metrics import DiceMetric, compute_meandice

import shutil
import tempfile
import time
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai import data
from monai.data import decollate_batch
from functools import partial

import torch

# Load JSON file
with open('/root/swinUNETR/jsons/dataset_validation.json', 'r') as f:
    test_files = json.load(f)
test_files = test_files["training"]
test_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
])

test_ds = Dataset(data=test_files, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming model and device setup are done elsewhere
model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=4,
    out_channels=3,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
).to(device)

model.load_state_dict(torch.load("/root/swinUNETR/pretrained_models/model.pt")["state_dict"])
model.to(device)
model.eval()

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
iou_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

# Initialize the model inference function
model_inferer = partial(sliding_window_inference, roi_size=[160, 160, 160], sw_batch_size=1, predictor=model, overlap=0.5)
iou_sum = 0
dice_sum = 0
# Compute metrics and visualize
for batch_data in test_loader:
    with torch.no_grad():
        # Prepare data
        images, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        # Model prediction
        outputs = model_inferer(images)
        outputs = torch.sigmoid(outputs)
        # Convert to one-hot format for comparison
        outputs = (outputs > 0.5).float()
        
        # Compute metrics
        dice_value = dice_metric(y_pred=outputs, y=labels)
        iou_value = iou_metric(y_pred=outputs, y=labels)
        
        print(f"Dice Score: {dice_value.item()}, IoU: {iou_value.item()}")
        dice_sum += dice_value.item()
        iou_sum += iou_value.item()

        
directory = "/root/dataset_val/MICCAI_BraTS2020_TrainingData"
subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
num_directories = len(subdirectories)

print(f"Dice Average: {dice_sum/num_directories}, IoU Average: {iou_sum/num_directories}")