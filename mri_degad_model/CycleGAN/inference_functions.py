import nibabel as nib 
import os
import glob
from PIL import Image
import numpy as np
import imageio
from torchvision import transforms
from compute_loss import compute_mae, compute_ssim
import torch
from generator_model import Generator
import config

class PadToSize:
    def __init__(self, size):
        self.target_width = size[0]
        self.target_height = size[1]

    def __call__(self, img):
        w, h = img.size
        pad_w = max(0, self.target_width - w)
        pad_h = max(0, self.target_height - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return transforms.functional.pad(img, padding, fill=-1, padding_mode='constant')

def get_test_pairs(input_dir):
    work_dir = os.path.join(input_dir, "work")
    subject_dirs = glob.glob(os.path.join(work_dir, "sub-*"))
    subjects = []
    
    for directory in subject_dirs:
        if os.path.isdir(directory): 
            subjects.append(directory)
    data_dicts = []
    for sub in subjects:   
        gad_images = glob.glob(os.path.join(sub, "ses-pre", "normalize", "*acq-gad*_T1w.nii.gz"))
        nogad_images = glob.glob(os.path.join(sub, "ses-pre", "normalize", "*acq-nongad*_T1w.nii.gz"))
        if gad_images and nogad_images:
            data_dicts.append({"image": gad_images[0], "label": nogad_images[0], "image_filepath": gad_images[0]})
    print("Loaded", len(data_dicts), "paired samples.")
    return data_dicts

def slice_gad_2d(gad_image_path, output_dir):
    img = nib.load(gad_image_path)
    data = img.get_fdata()
    print("shape:", data.shape)
    subject_id = os.path.basename(gad_image_path).split("_")[0]

    # 0, 1, 2 = sagittal, coronal, axial 
    # axial is the most common slice for ml

    for i in range(data.shape[2]):  
        slice_2d = data[:, :, i]

        min_pixel_val = np.min(slice_2d)
        max_pixel_val = np.max(slice_2d)

        # avoid divide-by-zero for blank slices
        if np.max(slice_2d) == np.min(slice_2d):
            continue

        slice_2d = ((slice_2d - min_pixel_val) / (max_pixel_val - min_pixel_val) * 255).astype(np.uint8)

        # converts to image and greyscale 
        img_pil = Image.fromarray(slice_2d).convert("L")
        
        subject_dir = os.path.join(output_dir, f"{subject_id}-2d_slices")
        os.makedirs(subject_dir, exist_ok=True)

        slice_filename = f"{subject_id}_slice_{i:03d}.png"
        print("saving image slice_filename")
        img_pil.save(os.path.join(subject_dir, slice_filename))
    return subject_dir, subject_id

def load_model(checkpoint_path, device):
    model = Generator(img_channels=1, num_features=64, num_residuals=9).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def apply_model(model_path, slice_dir, output_dir, subject_id, device):
    model = load_model(model_path, device)

    preprocess = transforms.Compose([
        PadToSize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    slice_files = sorted([
        f for f in os.listdir(slice_dir)
        if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])

    for fname in slice_files:
        path = os.path.join(slice_dir, fname)
        img = Image.open(path).convert("L")  # Ensure grayscale
        input_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dim

        with torch.no_grad():
            fake_nogad = model(input_tensor)
        
        # Output is tensor, shape [1,1,H,W]
        fake_nogad_img = fake_nogad.squeeze().cpu().numpy()
        
        # Rescale output to [0,255] uint8 for saving or stacking
        fake_nogad_img = (fake_nogad_img - fake_nogad_img.min()) / (fake_nogad_img.ptp() + 1e-5)
        fake_nogad_img = (fake_nogad_img * 255).astype(np.uint8)

        degad_dir = os.path.join(output_dir, f"{subject_id}-degad")
        os.makedirs(degad_dir, exist_ok=True)

        save_path = os.path.join(degad_dir, f"{fname}")
        Image.fromarray(fake_nogad_img).save(save_path)

        print("added line in file") 
    
    return degad_dir

def rebuild_3d(degad_dir, gad_image_path, subject_id):
    slice_files = sorted([
        f for f in os.listdir(degad_dir)
        if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])

    slices = []
    for file in slice_files:
        img = imageio.imread(os.path.join(degad_dir, file))
        if img.ndim == 3:
            img = img[:, :, 0]
        slices.append(img)

    volume = np.stack(slices, axis=0)
    gad_affine = np.eye(4)
    
    output_path = os.path.join(degad_dir, f"{subject_id}-reconstructed.nii.gz")

    nifti_img = nib.Nifti1Image(volume, gad_affine)
    nib.save(nifti_img, output_path)
    return output_path

# resamle / register to gad image with itksnap 

# only for testing
def compute_metrics(degad, nogad, sub):
    mae = compute_mae(degad, nogad)
    ssim = compute_ssim(degad, nogad)

    result_text = (
        f"Subject ID: {sub}\n"
        f"MAE: {mae:.6f}\n"
        f"SSIM: {ssim:.6f}\n"
        f"{'-'*30}\n"
    )

    output_txt = f"output_stats.txt"

    with open(output_txt, "a") as f:
        f.write(result_text)




    


