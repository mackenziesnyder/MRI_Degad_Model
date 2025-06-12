import glob 
import os
import numpy as np
from PIL import Image
import nibabel as nib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_gad_and_degad_nifti(input_dir):
    print("input_dir: ", input_dir)

    work_dir = os.path.join(input_dir, "work")
    subject_dirs = glob.glob(os.path.join(work_dir, "sub-*"))
    subjects = []
    
    for directory in subject_dirs:
        if os.path.isdir(directory): 
            subjects.append(directory)
    
    data_dicts = []

    for sub in subjects:
        print("sub: ", sub)   
        gad_images = glob.glob(os.path.join(sub, "ses-pre", "normalize", "*acq-gad*_T1w.nii.gz"))
        nogad_images = glob.glob(os.path.join(sub, "ses-pre", "normalize", "*acq-nongad*_T1w.nii.gz"))

        if gad_images and nogad_images:
            data_dicts.append({"train_a": gad_images[0], "train_b": nogad_images[0]})

    # if gad_images and nogad_images:
    #     data_dicts.append({"train_a": gad_images[0], "train_b": nogad_images[0]})
    print("Loaded", len(data_dicts), " total samples.")
    
    # 2 different datasets - one for gad, one for nogad

    gad_paths = [d["train_a"] for d in data_dicts]
    nogad_paths = [d["train_b"] for d in data_dicts]

    train_a, test_a = train_test_split(gad_paths, test_size=0.2, random_state=42)
    train_b, test_b = train_test_split(nogad_paths, test_size=0.2, random_state=42)
    
    print(f"Train: {len(train_a)}, Test: {len(test_a)}")
    print(f"Train: {len(train_b)}, Test: {len(test_b)}")

    return train_a, test_a, train_b, test_b

def is_informative(
    slice_2d,
    threshold_nonzero_ratio=0.01,
    center_crop_ratio=0.5,
    center_nonzero_thresh=0.05,
    side_balance_thresh=0.4,
    vertical_balance_thresh=0.4
):
    """
    Filters out MRI slices that are:
    - Mostly empty,
    - Missing central brain structures,
    - Heavily cut off on one side (scanner issue).
    """

    # --- Global non-zero check ---
    nonzero_ratio = np.count_nonzero(slice_2d) / slice_2d.size

    # --- Center content check ---
    h, w = slice_2d.shape
    ch, cw = int(h * center_crop_ratio), int(w * center_crop_ratio)
    start_h, start_w = (h - ch) // 2, (w - cw) // 2
    center_crop = slice_2d[start_h:start_h + ch, start_w:start_w + cw]
    center_nonzero_ratio = np.count_nonzero(center_crop) / center_crop.size

    # --- Left/right balance check ---
    left_half = slice_2d[:, :w // 2]
    right_half = slice_2d[:, w // 2:]

    left_nonzero = np.count_nonzero(left_half)
    right_nonzero = np.count_nonzero(right_half)

    if (left_nonzero + right_nonzero) == 0:
        side_balance = 1.0  # treat as completely unbalanced
    else:
        side_balance = abs(left_nonzero - right_nonzero) / (left_nonzero + right_nonzero)

    # --- top/bottom balance check ---
    top_half = slice_2d[:h // 2, :]
    bottom_half = slice_2d[h // 2:, :]
    
    top_nonzero = np.count_nonzero(top_half)
    bottom_nonzero = np.count_nonzero(bottom_half)

    if (top_nonzero + bottom_nonzero) == 0:
        vertical_balance = 1.0  # treat as completely unbalanced
    else:
        vertical_balance = abs(top_nonzero - bottom_nonzero) / (top_nonzero + bottom_nonzero)

    # --- Logging/debug ---
    if nonzero_ratio < threshold_nonzero_ratio:
        print("Filtered out: low global non-zero ratio")

    if center_nonzero_ratio < center_nonzero_thresh:
        print("Filtered out: low center content")

    if side_balance > side_balance_thresh:
        print(f"Filtered out: unbalanced sides (L:{left_nonzero}, R:{right_nonzero})")
    
    if vertical_balance > vertical_balance_thresh:
        print(f"Filtered out: unbalanced vertical (L:{top_nonzero}, R:{bottom_nonzero})")
    
    # is_valid = (
    #     nonzero_ratio > threshold_nonzero_ratio and
    #     center_nonzero_ratio > center_nonzero_thresh and
    #     side_balance <= side_balance_thresh and 
    #     vertical_balance <= vertical_balance_thresh
    # )

    # if not is_valid:
    #     plt.imshow(slice_2d, cmap='gray')
    #     plt.title("Filtered Slice")
    #     plt.axis("off")
    #     plt.show()

    return (
        nonzero_ratio > threshold_nonzero_ratio and
        center_nonzero_ratio > center_nonzero_thresh and
        side_balance <= side_balance_thresh and 
        vertical_balance <= vertical_balance_thresh
    )

def slice_3d_images(nifti_paths, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)

    for path in nifti_paths:
        img = nib.load(path)
        data = img.get_fdata()
        print("shape:", data.shape)
        subject_id = os.path.basename(path).split("_")[0]

        # do not slice edges 
        start_idx = 10
        end_idx = data.shape[2] - 30

        # 0, 1, 2 = sagittal, coronal, axial 
        # axial is the most common slice for ml

        for i in range(start_idx, end_idx):  
            slice_2d = data[:, :, i]

            min_pixel_val = np.min(slice_2d)
            max_pixel_val = np.max(slice_2d)

            # avoid divide-by-zero for blank slices
            if np.max(slice_2d) == np.min(slice_2d):
                continue
            
            
            if not is_informative(slice_2d):
                continue

            # in order to covert to image:
            # shifts minimum pixel value to 0
            # scales the values to be between 0 - 1
            # multiplies to be scaled between 0 - 255
            # converts to 8-bit unsigned to save the image 
            slice_2d = ((slice_2d - min_pixel_val) / (max_pixel_val - min_pixel_val) * 255).astype(np.uint8)

            # converts to image and greyscale 
            img_pil = Image.fromarray(slice_2d).convert("L")
            
            slice_filename = f"{tag}_{subject_id}_slice_{i:03d}.png"
            print("saving image slice_filename")
            img_pil.save(os.path.join(output_dir, slice_filename))

    print(f"Sliced and saved {tag} images to {output_dir}")








