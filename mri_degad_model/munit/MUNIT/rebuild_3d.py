import nibabel as nib 
import os
import imageio
import numpy as np
from pathlib import Path

def rebuild_3d(slice_dir):
    
    slice_files = sorted([
        f for f in os.listdir(slice_dir)
        if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])

    slices = []
    for file in slice_files:
        img = imageio.imread(os.path.join(slice_dir, file))
        if img.ndim == 3:
            img = img[:, :, 0]
        slices.append(img)

    volume = np.stack(slices, axis=0)  # Shape: (depth, height, width)
    return volume

def save_nifti(volume, output_path, affine=np.eye(4)):
    nifti_img = nib.Nifti1Image(volume, affine)
    nib.save(nifti_img, output_path)
    print(f"Saved 3D volume as NIfTI: {output_path}")

# === Example usage ===
home = Path.home()

slice_folder = home / "graham/scratch/2dtest"  # replace with your actual path
output_nifti = home / "graham/scratch/reconstructed_volume.nii.gz"

volume = rebuild_3d(slice_folder)
save_nifti(volume, output_nifti)


