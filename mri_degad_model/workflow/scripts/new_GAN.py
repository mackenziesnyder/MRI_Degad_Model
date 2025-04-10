import torch
import os
import glob
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
import torch.nn as nn
import time 
from monai.utils import progress_bar
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from pathlib import Path
import argparse

def train_GAN(input_dir,patch_size, batch_size,lr, filter_num_G, filter_num_D, depth_G, train_steps_d, loss_func, output_direct):
    
    output_dir = f"{output_direct}/patch-{patch_size}_batch-{batch_size}_LR-{lr}_filter_G-{filter_num_G}_filter_D-{filter_num_D}_depth_G-{depth_G}_train_steps_d_{train_steps_d}_loss_func_{loss_func}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subjects = sorted(glob.glob(os.path.join(input_dir, "sub-*")))
    data_dicts = []
    # create a dictonary of matching gad and nogad files
    for sub in subjects:
        gad_images = glob.glob(os.path.join(sub, "work", "ses-pre", "anat", "*_gad*_T1w.nii.gz"))
        nogad_images = glob.glob(os.path.join(sub, "work", "ses-pre", "anat", "*nongad*_T1w.nii.gz"))
        if gad_images and nogad_images:
            data_dicts.append({"image": gad_images[0], "label": nogad_images[0]})
    print("Loaded", len(data_dicts), "paired samples.")

    train, test = train_test_split(data_dicts, test_size=0.2, random_state=42)
    print(f"Train: {len(train)}, Test: {len(test)}")

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
    test_transforms = Compose([
        SpatialPadd(keys = ("image","label"),spatial_size = dims_tuple),
        ToTensord(keys=["image", "label"])
    ])

    train_ds = Dataset(data=train, transform=train_transforms)
    test_ds = Dataset(data=test, transform=test_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)

    # if depth is 3, and filter is 16, channels = 16, 32, 64
    channels = []
    for i in range(depth_G):
        channels.append(filter_num_G)
        filter_num_G *=2
    print("channels: ", channels)
    # if depth is 3, strides = 2, 2, 1 
    strides = []
    for i in range(depth_G - 1):
        strides.append(2)
    strides.append(1)
    print("strides: ", strides)
    # define model 
    gen_model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=channels,
        strides=strides,
        num_res_units=2,
        dropout=0.2,
        norm='BATCH'
    ).apply(monai.networks.normal_init).to(device)
    trainable_params_gen = sum(p.numel() for p in gen_model.parameters() if p.requires_grad)

    disc_model = GANDiscriminator(filter_num_D).apply(monai.networks.normal_init).to(device)
    trainable_params_disc = sum(p.numel() for p in disc_model.parameters() if p.requires_grad)

    learning_rate = lr
    betas = (0.5, 0.999)
    gen_opt = torch.optim.Adam(gen_model.parameters(), lr = learning_rate, betas=betas)
    disc_opt = torch.optim.Adam(disc_model.parameters(), lr = learning_rate, betas=betas)
    epoch_loss_values = [float('inf')] # list of generator  loss calculated at the end of each epoch
    disc_loss_values = [float('inf')] # list of discriminator loss values calculated at end of each epoch
    disc_train_steps = int(train_steps_d)# number of times to loop thru discriminator for each batch
    gen_training_steps = int(train_loader / batch_size) # batch_size is a tunable param
    disc_training_steps = disc_train_steps * gen_training_steps # number of validation steps per epoch
    max_epochs = 250
    loss = torch.nn.L1Loss().to(device)
    test_loss = 0

    start = time.time()

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        gen_model.train()
        disc_model.train()
        progress_bar(
            index = epoch + 1,
            count = max_epochs, 
            desc = f"epoch {epoch + 1}, avg gen loss: {epoch_loss_values[-1]:.4f}, avg disc loss: {disc_loss_values[-1]:.4f}",
            newline=True
        )
        average_train_loss_gen = 0
        average_train_loss_disc = 0 
        for i, batch in enumerate(train_loader):
            gad_images, nongad_images  =batch["image"].to(device), batch["label"].to(device)
            gen_opt.zero_grad()
            # apply generator model on gad images 
            degad_images = gen_model(gad_images)
            # apply discriminator model 
            disc_fake_pred = disc_model(torch.cat([gad_images, degad_images], dim=1)) # getting disc losses when fed fake images
            gen_loss = GeneratorLoss(nongad_images, degad_images, disc_fake_pred,device) # getting generator losses
            gen_loss.backward()# computes gradient(derivative) of current tensor, automatically frees part of greaph that creates loss
            gen_opt.step() # updates parameters to minimize loss
            average_train_loss_gen += gen_loss.item()
            for _ in range(disc_train_steps):
                gad_images, nongad_images = gad_images.clone().detach(), nongad_images.clone().detach() # need to recall it for each iteration to avoid error message of backpropagation through a graph a second time after gradients have been freed             
                degad_images = gen_model(gad_images) # feeding CNN with gad images             
                disc_opt.zero_grad() # resetting gradient for discrminator to 0           
                disc_real_pred = disc_model(torch.cat([gad_images, nongad_images], dim=1))
                disc_fake_pred = disc_model(torch.cat([gad_images, degad_images], dim=1))             
                disc_loss = DiscriminatorLoss(disc_real_pred,disc_fake_pred,device)
                disc_loss.backward() #initializes back propagation to compute gradient of current tensors 
                disc_opt.step() # updates parameters to minimize loss
                average_train_loss_disc += disc_loss.item() # taking sum of disc loss for the number of steps for this batch
        average_train_loss_gen /= gen_training_steps # epoch loss is the total loss by the end of that epoch divided by the number of steps
        epoch_loss_values.append(average_train_loss_gen) #updates the loss value for that epoch
        average_train_loss_disc /= disc_training_steps# average disc epoch loss is the total loss divided by the number of discriminator steps
        disc_loss_values.append(average_train_loss_disc) # av
        gen_model.eval()
    torch.save(gen_model.state_dict(), f"{output_dir}/trained_generator.pt")
    torch.save(disc_model.state_dict(), f"{output_dir}/trained_discriminator.pt")
    end = time.time()
    time = end - start
    print("time for training: ", time)

    with open (f'{output_dir}/model_stats.txt', 'w') as file:  
        file.write(f'Training time: {time}\n') 
        file.write(f'Number of trainable parameters in generator: {trainable_params_gen}\n')
        file.write(f'Number of trainable parameters in discriminator: {trainable_params_disc}\n')
        file.write(f'generator loss: {epoch_loss_values[-1]} discriminator loss: {disc_loss_values[-1]}')
    plt.figure(figsize=(12,5))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(list(range(len(epoch_loss_values))), epoch_loss_values, label="Generator Loss")
    plt.plot(list(range(len(disc_loss_values))), disc_loss_values , label="Discriminator Loss")
    plt.grid(True, "both", "both")
    plt.title("Generator and Discriminator Loss")
    plt.legend()
    plt.savefig(f'{output_dir}/lossfunction.png')
    plt.close()

    gen_model.load_state_dict(torch.load(f'{output_dir}/trained_generator.pt'))
    gen_model.eval()
    output_dir_test = Path(output_dir) / "test"
    output_dir_test.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            gad_images, nogad_images = batch["image"].to(device), batch["label"].to(device)
            # Sliding window inference for large 3D volumes
            degad_images = sliding_window_inference(gad_images, patch_size, 1, gen_model)
            loss_value = loss(degad_images, nogad_images)
            test_loss += loss_value.item()
            # Save degad images as NIfTI
            for i in range(degad_images.shape[0]):
                gad_path = batch["image_meta_dict"]["filename_or_obj"][i]
                gad_nib = nib.load(gad_path)
                sub = Path(gad_path).name.split("_")[0]
                degad_name = f"{sub}_acq-degad_T1w.nii.gz"
                degad_nib = nib.Nifti1Image(
                    degad_images[i, 0].detach().numpy()*100,
                    affine=gad_nib.affine,
                    header=gad_nib.header,
                )
                os.makedirs(f'{output_dir_test}/bids/{sub}/ses-pre/anat', exist_ok=True) # save in bids format
                output_path = f'{output_dir_test}/bids/{sub}/ses-pre/anat/{degad_name}'
                nib.save(degad_nib, output_path)
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

class GANDiscriminator(nn.Module):
    def __init__(self, ini_filters):
        super().__init__()
        in_channels=2
        kernel_size=3
       
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, ini_filters, kernel_size, stride=2, padding=1),
            nn.InstanceNorm3d(ini_filters),
            nn.PReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(ini_filters, ini_filters*2, kernel_size, stride=2, padding=1),
            nn.InstanceNorm3d(ini_filters*2),
            nn.PReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(ini_filters*2, ini_filters*4, kernel_size, stride=2, padding=1),
            nn.InstanceNorm3d(ini_filters*4),
            nn.PReLU()
        )
        
        self.conv_out = nn.Conv3d(ini_filters*4, 1, kernel_size, stride=1, padding=0)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.tanh(x)
        return x

def GeneratorLoss(nongad_images,degad_images, fake_preds, device):
    """
    Loss function is the sum of the binary cross entropy between the error of the discriminator output btwn gad and degad (fake prediction) and the root mean square error betwn as nongad and degad images multiplies by scalar weight coefficient
    nongad_image= real nongad images from the sample
    degad_images= generated nongad images from the generator
    fake_preds: output of discriminator when fed fake data
    """
    
    coeff = 0.01
    
    BCE_loss= torch.nn.BCELoss() 
    real_target = torch.ones((fake_preds.shape[0], fake_preds.shape[1], fake_preds.shape[2], fake_preds.shape[3], fake_preds.shape[4])) #new_full returns a tensor filled with 1 with the same shape as the discrminator prediction 
    fake_preds = torch.sigmoid(fake_preds) # applying sigmmoid function to output of the discriminator to map probability between 0 and 1
    BCE_fake = BCE_loss(fake_preds.to(device), real_target.to(device)) # BCE loss btwn the output of discrim when fed fake data and 1 <- generator wants to minimize this
    L1_loss = torch.nn.L1Loss()
    loss = L1_loss(degad_images, nongad_images)  # producing RMSE between ground truth nongad and degad
    generator_loss = loss*coeff + BCE_fake
    return generator_loss

def DiscriminatorLoss(real_preds, fake_preds,device):
    """
    Loss function for the discriminator: The discriminator loss is calculated by taking the sum of the L2 error of the discriminator output btwn gad and nongad( real prediction ) and the L2 error of the output btwn gad and degad( fake predition)
    
    real_preds: output of discriminator when fed real data
    fake_preds: output of discriminator when fed fake data
    """
    
    real_target = torch.ones((real_preds.shape[0], real_preds.shape[1], real_preds.shape[2],real_preds.shape[3], real_preds.shape[4])) #new_full returns a tensor filled with 1 with the same shape as the discrminator prediction 
    
    fake_target = torch.zeros((fake_preds.shape[0], fake_preds.shape[1], fake_preds.shape[2], fake_preds.shape[3], fake_preds.shape[4])) #new_full returns a tensor filled with 0 w/ the same shape as the generator prediction
    BCE_loss =  torch.nn.BCELoss().to(device)  # creates a losss value for each batch, averaging the value across all elements
    # Apply sigmoid to discriminator outputs, to fit between 0 and 1
    real_preds = torch.sigmoid(real_preds).cuda()
    fake_preds = torch.sigmoid(fake_preds).cuda()
    
    BCE_fake = BCE_loss(fake_preds.cuda(), fake_target.cuda()) # BCE loss btwn the output of discrim when fed fake data and 0 <- generator wants to minimize this
    BCE_real = BCE_loss(real_preds.cuda(), real_target.cuda()) # BCE loss btwn the output of discrim when fed real data and 1 <- generator wants to minimize this
    
    return BCE_real + BCE_fake   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN degad model with specified parameters.")
    parser.add_argument("--input", nargs='+', required=True, help="Path to the training and validation data, in that order")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size for training.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("--filterG", type=int, required=True, help="Number of filters in initial layer of generator.")
    parser.add_argument("--filterD", type=int, required=True, help="Number of filters in initial layer of discriminator.")
    parser.add_argument("--depthG", type=int, required=True, help="Depth of generator.")
    parser.add_argument("--loss", required=True, help="Type of loss function to apply: mae, ssim or both.")
    parser.add_argument("--output_dir", required=True, help="Output directory for model to be saved in.")
    parser.add_argument("--trainD", required=True, help="Number of steps being applied to discriminator.")
    args = parser.parse_args()
    input_dir = args.input
    patch_size = args.patch_size
    batch_size = args.batch_size
    lr = args.lr
    filter_num_G = args.filterG
    filter_num_D = args.filterD
    depth_G = args.depthG
    train_steps_d = args.trainD
    loss_func=args.loss
    output_direct=args.output_dir
    train_GAN(input_dir,patch_size, batch_size,lr,filter_num_G, filter_num_D, depth_G, train_steps_d, loss_func, output_direct)