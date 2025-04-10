import torch 
import glob 
import os
import nibabel as nib
from sklearn.model_selection import train_test_split
import monai
from monai.transforms import (
    Compose,
    Rand3DElasticd,
    SpatialPadd,
    RandFlipd,
    RandSpatialCropd,
    ToTensord
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
import time
from pytorchtools import EarlyStopping
from monai.utils import progress_bar
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from pathlib import Path
import argparse

def train_CNN(input_dir, patch_size, batch_size, lr, filter, depth, loss_func, output_direct):
    
    output_dir = f"{output_direct}/patch-{patch_size}_batch-{batch_size}_LR-{lr}_filter-{filter}_depth-{depth}_loss-{loss_func}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # subject folders 
    subjects = sorted(glob.glob(os.path.join(input_dir, "work", "sub-*")))
    print("subjects: ", subjects)
    data_dicts = []
    # create a dictonary of matching gad and nogad files
    for sub in subjects:
        print("sub: ", sub)
        gad_images = glob.glob(os.path.join(sub, "ses-pre", "normalize", "*acq-nongad*_T1w.nii.gz"))
        print("gad imag", gad_images)
        nogad_images = glob.glob(os.path.join(sub, "ses-pre", "normalize","*acq-nongad*_T1w.nii.gz"))
        if gad_images and nogad_images:
            data_dicts.append({"image": gad_images[0], "label": nogad_images[0]})
    print("Loaded", len(data_dicts), "paired samples.")

    train_val, test = train_test_split(data_dicts, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.176, random_state=42) # 0.176 ≈ 15% of the full data
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # set size of image to patch size (patch_size, patch_size, patch_size)
    dims_tuple = (patch_size,)*3
    # want to train with patches
    train_transforms = Compose([
        SpatialPadd(keys = ("image","label"), spatial_size = dims_tuple), #ensures all data is around the same size
        Rand3DElasticd(keys = ("image","label"), sigma_range = (0.5,1), magnitude_range = (0.1, 0.4), prob=0.4, shear_range=(0.1, -0.05, 0.0, 0.0, 0.0, 0.0), scale_range=0.5, padding_mode= "zeros"),
        RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=1),
        RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=0),
        RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=2),
        RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_center=True, random_size=False),
        ToTensord(keys=["image", "label"])
    ])
    # want to validate and test with whole images 
    val_transforms = Compose([
        SpatialPadd(keys = ("image","label"),spatial_size = dims_tuple),
        ToTensord(keys=["image", "label"])
    ])

    train_ds = Dataset(data=train, transform=train_transforms)
    val_ds = Dataset(data=val, transform=val_transforms)
    test_ds = Dataset(data=test, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    channels = []
    for i in range(depth):
        channels.append(filter)
        filter *=2
    print("channels: ", channels)
    strides = []
    for i in range(depth - 1):
        strides.append(2)
    strides.append(1)
    print("strides: ", strides)
    # define model 
    model = UNet(
        dimensions=3,
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
    patience = 22 # epochs it will take for training to terminate if no improvement
    early_stopping = EarlyStopping(patience=patience, verbose=True, path = f'{output_dir}/checkpoint.pt')
    max_epochs = 800
    loss = torch.nn.L1Loss().to(device)
    train_losses = [float('inf')]
    val_losses = [float('inf')]
    test_loss = 0

    start = time.time()
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        model.train()
        progress_bar(
            index=epoch+1, # displays what step we are of current epoch, our epoch number, training  loss
            count = max_epochs, 
            desc= f"epoch {epoch + 1}, training mae loss: {train_losses[-1]:.4f}, validation mae metric: {val_losses[-1]:.4f}",
            newline = True) # progress bar to display current stage in training
        # training
        avg_train_loss = 0
        for i, batch in enumerate(train_loader):
            # image is gad image, label is nogad image
            gad_images, nogad_images = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad() #resets optimizer to 0
            degad_images = model(gad_images)
            train_loss = loss(degad_images, nogad_images)
            train_loss.backward() # computes gradients for each parameter based on loss
            optimizer.step() # updates the model weights using the gradient
            avg_train_loss += train_loss.item() 
        avg_train_loss /= len(train_loader) # average loss per current epoch 
        train_losses.append(avg_train_loss) # append total epoch loss divided by the number of training steps in epoch to loss list
        model.eval()
        # validation 
        with torch.no_grad(): #we do not update weights/biases in validation training, only used to assess current state of model
            avg_val_loss = 0 # will hold sum of all validation losses in epoch and then average
            for i, batch in enumerate(val_loader): # iterating through dataloader
                gad_images, nogad_images = batch["image"].to(device), batch["label"].to(device)
                degad_images = model(gad_images)  
                val_loss = loss(degad_images, nogad_images)
                avg_val_loss += val_loss.item() 
            avg_val_loss /= len(val_loader) #producing average val loss for this epoch
            val_losses.append(avg_val_loss) 
            early_stopping(avg_val_loss, model) # early stopping keeps track of last best model
        if early_stopping.early_stop: # stops early if validation loss has not improved for {patience} number of epochs
            print("Early stopping, saving model") 
            break
    end = time.time()
    time = end - start
    print("time for training and validation: ", time)

    with open (f'{output_dir}/model_stats.txt', 'w') as file:  
        file.write(f'Training time: {time}\n') 
        file.write(f'Number of trainable parameters: {trainable_params}\n')
        file.write(f'Training loss: {train_losses[-patience]} \nValidation loss: {early_stopping.val_loss_min}')
        
    plt.figure(figsize=(12,5))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(list(range(len(train_losses))), train_losses, label="Training Loss")
    plt.plot(list(range(len(val_losses))),val_losses , label="Validation Loss")
    plt.grid(True, "both", "both")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f'{output_dir}/lossfunction.png')
    plt.close()

    model.load_state_dict(torch.load(f'{output_dir}/checkpoint.pt'))
    model.eval()
    output_dir_test = Path(output_dir) / "test"
    output_dir_test.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            gad_images, nogad_images = batch["image"].to(device), batch["label"].to(device)
            degad_images = sliding_window_inference(gad_images, patch_size, 1, model)
            loss_value = loss(degad_images, nogad_images)
            test_loss += loss_value.item()
            # to save the output files 
            # shape[0] gives number of images 
            for i in range(degad_images.shape[0]):
                gad_path = batch["image_meta_dict"]["filename_or_obj"][i]
                gad_nib = nib.load(gad_path)
                sub = Path(gad_path).name.split("_")[0] 
                degad_name = f"{sub}_acq-degad_T1w.nii.gz"             
                degad_nib = nib.Nifti1Image(
                    degad_images[i, 0].detach().numpy()*100, 
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
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size for training.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("--filter", type=int, required=True, help="Number of filters in initial layer.")
    parser.add_argument("--depth", type=int, required=True, help="Depth of U-Net.")
    parser.add_argument("--loss", required=True, help="Type of loss function to apply: mae, ssim or both.")
    parser.add_argument("--output_dir", required=True, help="Output directory for model to be saved in.")
    args = parser.parse_args()
    input_dir = args.input
    patch_size = args.patch_size
    batch_size = args.batch_size
    lr = args.lr
    filter_num=args.filter
    depth= args.depth
    loss_func=args.loss
    output_direct=args.output_dir
    train_CNN(input_dir,patch_size, batch_size,lr,filter_num,depth, loss_func, output_direct)