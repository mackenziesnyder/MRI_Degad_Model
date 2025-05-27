import os
import nibabel as nib
import numpy as np
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    CenterSpatialCrop
)

# --- Parameters ---
input_path = "/local/scratch/MRI_Degad_Model/model_qc/sub-P003_ses-pre_acq-nongad_run-01_desc-normalized_zscore_T1w.nii.gz"
output_path = "output.nii.gz"
crop_size = (256, 256, 256)

# --- Load and preprocess image ---
loader = LoadImage(image_only=True)
image = loader(input_path)  # np.ndarray
image = EnsureChannelFirst()(image)  # shape: (1, D, H, W)

# --- Center crop ---
cropper = CenterSpatialCrop(roi_size=crop_size)
cropped = cropper(image)  # shape: (1, 256, 256, 256)

# --- Save as .nii.gz ---
# Remove channel dimension before saving
cropped_np = cropped[0].numpy().astype(np.float32)

# Load original affine for correct orientation
original_img = nib.load(input_path)
affine = original_img.affine

cropped_img = nib.Nifti1Image(cropped_np, affine)
cropped_img.set_sform(None)
cropped_img.set_qform(None)

# Save cropped image
nib.save(cropped_img, output_path)
print(f"Cropped image saved to: {output_path}")
