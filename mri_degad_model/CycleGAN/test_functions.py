import nibabel as nib 
import os
import glob
from PIL import Image
import numpy as np
import imageio
from torchvision import transforms
from compute_loss import compute_mse, compute_mae, compute_dice, compute_ssim
import torch
from generator_model import Generator
import config
from pathlib import Path

class PadToSize:
    def __init__(self, size):
        self.target_width = size[0]
        self.target_height = size[1]

    def __call__(self, img):
        w, h = img.size
        pad_w = max(0, self.target_width - w)
        pad_h = max(0, self.target_height - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
        
def load_generator(checkpoint_path, device):
    model = Generator(img_channels=1, num_features=64, num_residuals=9).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def load_gad_nogad_pairs(test_dir):
    print("test_dir",test_dir)
    
    # creates dictionary of matching image and label paths
    cycle_test_dir = os.path.join(test_dir, "cyclegan_test")
    subject_dirs = glob.glob(os.path.join(cycle_test_dir, "sub-*"))
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

def slice_image(img_path, gad_slices_output_dir, tag):
    img = nib.load(img_path)
    data = img.get_fdata()
    print("shape:", data.shape)
    subject_id = os.path.basename(img_path).split("_")[0]

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

        # in order to covert to image:
        # shifts minimum pixel value to 0
        # scales the values to be between 0 - 1
        # multiplies to be scaled between 0 - 255
        # converts to 8-bit unsigned to save the image 
        slice_2d = ((slice_2d - min_pixel_val) / (max_pixel_val - min_pixel_val) * 255).astype(np.uint8)

        # converts to image and greyscale 
        img_pil = Image.fromarray(slice_2d).convert("L")

        # Ensure gad_slices_output_dir is a Path object
        gad_slices_output_dir = Path(gad_slices_output_dir)

        # Create output directory
        slice_dir = gad_slices_output_dir / f"slices-{tag}"
        slice_dir.mkdir(parents=True, exist_ok=True)
        
        slice_filename = os.path.join(slice_dir, f"{subject_id}_slice_{i:03d}.png")
        print("saving image slice_filename")
        img_pil.save(os.path.join(gad_slices_output_dir, slice_filename))

        return gad_slices_output_dir

def apply_model(slice_dir, output_dir):
    model_path = "genz.pth.tar"
    device = config.DEVICE
    model = load_generator(model_path, device)

    preprocess = transforms.Compose([
        PadToSize((256, 256)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    degad_dir = os.path.join(output_dir, "slices-degad")
    os.makedirs(degad_dir, exist_ok=True)

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

        save_path = os.path.join(degad_dir, f"degad_{fname}")
        Image.fromarray(fake_nogad_img).save(save_path)

        print("added line in file")

    return degad_dir

def rebuild_3d(slice_dir, output_path, tag):
    
    print("slice_dir: ", slice_dir)
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

    volume = np.stack(slices, axis=0)
    volume_rotated = np.transpose(volume, (1, 2, 0))
    affine = np.diag([1, 1, 1, 1])
    file_name = f"{tag}"

    nifti_img = nib.Nifti1Image(volume_rotated, affine)
    nib.save(nifti_img, file_name)
    print(f"Saved 3D volume as NIfTI: {file_name}")
    return file_name

def compute_image_similarity(degad_img, nogad_img):
    
    # Load the NIfTI image objects
    read_degad_img = nib.load(degad_img)
    read_nogad_img = nib.load(nogad_img)

    # Convert the objects to NumPy arrays
    degad = read_degad_img.get_fdata()
    nogad = read_nogad_img.get_fdata()

    # compute mae
    mae = compute_mae(degad, nogad)
    # compute mse 
    mse = compute_mse(degad, nogad)
    # compute SSIM
    ssim = compute_ssim(degad, nogad)
    # compute dice score 
    # dice = compute_dice(degad, nogad)

    subject_id = os.path.basename(degad_img).split("_")[0]

    # Format the output text
    result_text = (
        f"Subject ID: {subject_id}\n"
        f"MAE: {mae:.6f}\n"
        f"MSE: {mse:.6f}\n"
        f"SSIM: {ssim:.6f}\n"
        # f"Dice: {dice:.6f}\n"
        f"{'-'*30}\n"
    )

    output_txt = f"{subject_id}_output_stats.txt"
    # Append results to the text file
    with open(output_txt, "a") as f:
        f.write(result_text)



    

