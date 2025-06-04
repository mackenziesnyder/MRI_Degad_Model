import glob 
import os
import numpy as np
from PIL import Image
import nibabel as nib
from sklearn.model_selection import train_test_split

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

def slice_3d_images(nifti_paths, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)

    for path in nifti_paths:
        img = nib.load(path)
        data = img.get_fdata()
        print("shape:", data.shape)
        subject_id = os.path.basename(path).split("_")[0]

        # 0, 1, 2 = sagittal, coronal, axial 
        # axial is the most common slice for ml
        
        for i in range(data.shape[2]):  
            slice_2d = data[:, :, i]

            min_pixel_val = np.min(slice_2d)
            max_pixel_val = np.max(slice_2d)

            # avoid divide-by-zero for blank slices
            if np.max(slice_2d) == np.min(slice_2d):
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
            img_pil.save(os.path.join(output_dir, slice_filename))

    print(f"Sliced and saved {tag} images to {output_dir}")



