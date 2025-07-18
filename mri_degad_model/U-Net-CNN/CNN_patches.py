import torch 
import glob 
import os
import nibabel as nib
from sklearn.model_selection import train_test_split

import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Rand3DElasticd,
    ScaleIntensityd,
    Orientationd,
    Spacingd,
    ResizeWithPadOrCropd,
    RandFlipd,
    ToTensord,
    MapTransform,
    GridPatchd,
)
from monai.data import Dataset, DataLoader, PatchDataset
from monai.networks.nets import UNet
from monai.losses import SSIMLoss, PerceptualLoss

import time
import numpy as np

from monai.utils import progress_bar

import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from pathlib import Path
import argparse

class SaveImagePath(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        
    def __call__(self, data):
        data['image_filepath'] = data['image']
        return data
    
def train_CNN(input_dir, image_size, patch_size, batch_size, lr, filter, depth, loss_func, output_direct):
    
    output_dir = f"{output_direct}/image-{image_size}_batch-{batch_size}_LR-{lr}_filter-{filter}_depth-{depth}_loss-{loss_func}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_dir = input_dir[0]
    # creates dictionary of matching image and label paths
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
    
    # split into train, val, test
    train_val, test = train_test_split(data_dicts, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42) # 0.176 = 15% of the full data
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # set size of image
    dims_tuple_image = (image_size,)*3
    dims_tuple_patches = (patch_size,)*3
    print("dims_tuple_image: ", dims_tuple_image)
    print("dims_tuple_patches: ", dims_tuple_patches)

    # train tranforms 
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.0, 1.0, 1.0), 
            mode=("bilinear", "nearest")
        ),
        ResizeWithPadOrCropd(
            keys=["image", "label"], 
            spatial_size=(256, 256, 256)
        ),
        ScaleIntensityd(keys=["image"]),
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(0.5, 1),
            magnitude_range=(0.1, 0.3),
            prob=0.2,
            padding_mode="zeros"
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
        GridPatchd(
            keys=("image", "label"),
            patch_size=(dims_tuple_patches),
            offset=(0, 0, 0),
            stride=(dims_tuple_patches) # Non-overlapping
        ),
        ToTensord(keys=["image", "label"])
    ])

    sample_train = train_transforms(train[0])
    print("Test image shape:", sample_train["image"].shape)
    print("Test label shape:", sample_train["label"].shape)

    # validate 
    val_transforms = Compose([
        SaveImagePath(keys=["image"]),
        LoadImaged(keys=["image", "label"]),  # load image
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(
            keys=["image", "label"], 
            spatial_size=(256, 256, 256)
        ),
        ToTensord(keys=["image", "label"])
    ])

    train_ds = Dataset(data=train, transform=train_transforms)
    val_ds = Dataset(data=val, transform=val_transforms)
    test_ds = Dataset(data=test, transform=val_transforms)

    # training, validating, testing of whole data so use a batch size of 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=1, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=1, pin_memory=pin_memory)
    
    channels = []
    for i in range(depth):
        channels.append(filter)
        filter *=2
    print("channels: ", channels)
    strides = []
    for i in range(depth-1):
        strides.append(2)
    print("strides: ", strides)

    # define model 
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
        num_res_units=2,
        dropout=0.2,
        norm='BATCH'
    ).apply(monai.networks.normal_init).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    learning_rate = float(lr)
    # common defaults
    betas = (0.5, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=betas)
    best_model_path = f"{output_dir}/best_model.pt"
    max_epochs = 250
    patience = 10

    l1_loss = torch.nn.L1Loss().to(device) # mae
    ssim_loss = SSIMLoss(spatial_dims=3).to(device) #SSIM
    p_loss = PerceptualLoss(spatial_dims=3,network_type="vgg").to(device)

    train_losses = [float('inf')]
    val_losses = [float('inf')]
    best_val_loss = float('inf')
    test_loss = 0

    #loops through each individual image and its corresponding patch

    start = time.time()

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        model.train()

        train_loss_display = f"{train_losses[-1]:.4f}" if train_losses else "N/A"
        val_loss_display = f"{val_losses[-1]:.4f}" if val_losses else "N/A"

        progress_bar(
            index=epoch + 1,
            count=max_epochs,
            desc=f"epoch {epoch + 1}, training mae loss: {train_loss_display}, validation mae metric: {val_loss_display}",
            newline=True
        )

        # training
        avg_train_loss = 0
        patch_count = 0
        epochs_without_improvement = 0

        print("--------Training--------")
        
        # Iterate over the batch dimension (16 images in this case)
        for batch_idx, batch in enumerate(train_loader):
            gad_images, nogad_images = batch["image"].to(device), batch["label"].to(device)
            print(f"batch: {batch_idx + 1}")
            print(f"Original gad_images shape: {gad_images.shape}")  # e.g. [batch_size, num_patches, C, D, H, W]
            
            batch_size, num_patches, C, D, H, W = gad_images.shape
            
            # Flatten batch and patch dims into batch dimension
            gad_images = gad_images.view(-1, C, D, H, W)
            nogad_images = nogad_images.view(-1, C, D, H, W)
            
            print(f"Reshaped gad_images shape (patch batch): {gad_images.shape}")  # [batch_size*num_patches, C, D, H, W]
            print(f"Reshaped nogad_images shape (patch batch): {nogad_images.shape}")
            
            optimizer.zero_grad()
            degad_images = model(gad_images)
            print(f"Model output shape: {degad_images.shape}")

            mloss = l1_loss(degad_images, nogad_images)
            sloss = ssim_loss(degad_images, nogad_images)
            ploss = p_loss(degad_images, nogad_images)

            loss_value = (0.5 * mloss) + (0.3 * sloss) + (0.2 * ploss)
            loss_value.backward()
            optimizer.step()

            avg_train_loss += loss_value.item()
            patch_count += batch_size * num_patches

        avg_train_loss /= patch_count  # Average loss per epoch
        train_losses.append(avg_train_loss)  # Append to training losses list
        model.eval()
        
        print("--------Validation--------")
        # validation 
        with torch.no_grad():  # Do not update weights in validation
            avg_val_loss = 0  # Will hold sum of all validation losses
            for batch in val_loader:
                gad_images, nogad_images = batch["image"].to(device), batch["label"].to(device)
                degad_images = model(gad_images)

                mloss = l1_loss(degad_images, nogad_images)
                sloss = ssim_loss(degad_images, nogad_images)
                ploss = p_loss(degad_images, nogad_images)

                val_loss = (0.5 * mloss) + (0.3 * sloss) + (0.2 * ploss)
                avg_val_loss += val_loss.item() 
            
            avg_val_loss /= len(val_loader)  # Average validation loss for the epoch
            val_losses.append(avg_val_loss) 

            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model.")
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"No improvement in validation loss. Best remains {best_val_loss:.4f}.")

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    end = time.time()
    total_time = end - start
    print("Time for training and validation: ", total_time)
    
    with open (f'{output_dir}/model_stats.txt', 'w') as file:  
        file.write(f'Training time: {total_time:.2f} seconds\n') 
        file.write(f'Number of trainable parameters: {trainable_params}\n')

        if len(train_losses) > patience:
            file.write(f'Training loss (epoch {-patience}): {train_losses[-patience]:.4f}\n')
        else:
            file.write(f'Training loss (last epoch): {train_losses[-1]:.4f}\n')

        file.write(f'Validation loss (best): {best_val_loss:.4f}\n')
        
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lossfunction.png')
    plt.close()

    model.load_state_dict(torch.load(f'{output_dir}/best_model.pt', map_location=torch.device('cpu')))
    model.eval()

    output_dir_test = Path(output_dir) / "test"
    output_dir_test.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):      
            gad_images, nogad_images = batch["image"].to(device), batch["label"].to(device)
            gad_paths = batch["image_filepath"]
            degad_images = sliding_window_inference(gad_images, image_size, 1, model)
            degad_images = degad_images[:, :, :image_size, :image_size, :image_size]

            mloss = l1_loss(degad_images, nogad_images)
            sloss = ssim_loss(degad_images, nogad_images)
            ploss = p_loss(degad_images, nogad_images)

            loss_value = (0.5 * mloss) + (0.3 * sloss) + (0.2 * ploss)

            test_loss += loss_value.item()

            # to save the output files 
            # shape[0] gives number of images 
            for j in range(degad_images.shape[0]):
                gad_path = gad_paths[j] # test dictionary image file name
                print(gad_path)
                gad_nib = nib.load(gad_path)
                sub = Path(gad_path).name.split("_")[0]
                degad_name = f"{sub}_acq-degad_T1w.nii.gz"
                
                degad_nib = nib.Nifti1Image(
                    degad_images[j, 0].detach().numpy()*100, 
                    affine=gad_nib.affine,
                    header=gad_nib.header
                )

                os.makedirs(f'{output_dir_test}/bids/{sub}/ses-pre/anat', exist_ok=True) # save in bids format
                output_path = f'{output_dir_test}/bids/{sub}/ses-pre/anat/{degad_name}'
                nib.save(degad_nib, output_path)
        
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN degad model with specified parameters.")
    parser.add_argument("--input", nargs='+', required=True, help="Path to the training and validation data, in that order")
    parser.add_argument("--image_size", type=int, required=True, help="image size for training.")
    parser.add_argument("--patch_size", type=int, required=True, help="patch size for training.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("--filter", type=int, required=True, help="Number of filters in initial layer.")
    parser.add_argument("--depth", type=int, required=True, help="Depth of U-Net.")
    parser.add_argument("--loss", required=True, help="Type of loss function to apply: mae, ssim or both.")
    parser.add_argument("--output_dir", required=True, help="Output directory for model to be saved in.")
    args = parser.parse_args()
    input_dir = args.input
    image_size = args.image_size
    patch_size = args.patch_size
    batch_size = args.batch_size
    lr = args.lr
    filter_num=args.filter
    depth= args.depth
    loss_func=args.loss
    output_direct=args.output_dir
    train_CNN(input_dir,image_size, patch_size, batch_size,lr,filter_num,depth, loss_func, output_direct)