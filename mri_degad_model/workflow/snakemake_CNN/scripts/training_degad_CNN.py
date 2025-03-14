#!/usr/bin/env python
# coding: utf-8

# In[1]:


import monai
import shutil
from monai.transforms import (
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    Spacingd,
    RandWeightedCrop,
    RandRotate,
    RandFlip,
    Rand3DElasticd,
    Rand3DElastic,
    RandRotated,
    LoadImage,
    EnsureChannelFirstd,
    Orientationd,
    EnsureChannelFirst,
    ScaleIntensityd,
    RandFlip,
    ToTensor,
    SpatialPadd,
    ToTensord,
    ScaleIntensity,
    RandFlipd)
import nibabel
import shutil
import tqdm
from torchmetrics import MeanSquaredError
import time
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, Dataset,PatchDataset, PersistentDataset, SmartCacheDataset, ThreadDataLoader
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import nibabel
from glob import glob
from monai.networks.blocks import Convolution
from monai.networks.nets import Discriminator, Generator
from monai.utils import progress_bar
import torch.nn as nn
import torchmetrics 
from pytorchtools import EarlyStopping
import numpy 
import torchvision.transforms as transforms
import random
from functools import reduce
from operator import mul
from torch.utils.data import DataLoader

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM


import argparse

def train_model(input_files, patch_size,batch_size, lr,filter_num,depth,num_conv, loss_func):
    
    print(f"Training degad CNN with input files: {input_files[0]} {input_files[1]}\npatch_size: {patch_size}\nbatch size: {batch_size}\nlearning rate: {lr}\nnumber of initial filters: {filter_num}\nCNN depth: {depth}\nnumber of convolutions per block: {num_conv}\nloss: {loss_func}")

    output_dir = f"/project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/snakemake_CNN/output/patch-{patch_size}_batch-{batch_size}_LR-{lr}_filter-{filter_num}_depth-{depth}_convs-{num_conv}_loss-{loss_func}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    fname_tr=input_files[0]# training file
    radius_actual = [int(patch_size/2-1)]*3 # getting c3d patch radius ie. if 32 ^3 patch size, it is 15
    patch_radius= numpy.array(radius_actual) # Patch dimensions
    dims = 1+2*patch_radius # numpyt
    dims_tuple = (patch_size,)*3
    k = 2  # Number of channels
    bps = (4 * k * numpy.prod(dims)) # Bytes per sample
    np_tr = os.path.getsize(fname_tr) // bps  # Number of samples
    arr_shape_tr= (int(np_tr),dims[0],dims[1],dims[2], k)
    arr_train = numpy.memmap(fname_tr,'float32','r+',shape=arr_shape_tr)
    
    fname_va=input_files[1] # validation file   
    np_va = os.path.getsize(fname_va) // bps      # Number of samples
    arr_shape_va= (int(np_va),dims[0],dims[1],dims[2], k)
    arr_val= numpy.memmap(fname_va,'float32','r+',shape=arr_shape_va)

    arr_train = numpy.swapaxes(arr_train,1,4)
    arr_val = numpy.swapaxes(arr_val,1,4)

    train_size=int(arr_train.shape[0])
    val_size=int(arr_val.shape[0])
    arr_train_image = arr_train[0:train_size,0,:,:,:].reshape(train_size,1,arr_train.shape[2],arr_train.shape[3],arr_train.shape[4])
    arr_train_label = arr_train[0:train_size,1,:,:,:].reshape(train_size,1,arr_train.shape[2],arr_train.shape[3],arr_train.shape[4])

    arr_val_image = arr_val[0:val_size,0,:,:,:].reshape(val_size,1, arr_val.shape[2],arr_val.shape[3],arr_val.shape[4])
    arr_val_label = arr_val[0:val_size,1,:,:,:].reshape(val_size,1, arr_val.shape[2],arr_val.shape[3],arr_val.shape[4])
    arr_train_dict= [{"image": gad_name, "label": nongad_name} for gad_name, nongad_name in zip(arr_train_image,arr_train_label)]
    arr_val_dict= [{"image": gad_name, "label": nongad_name} for gad_name, nongad_name in zip(arr_val_image,arr_val_label)]
    
    train_transforms = Compose([SpatialPadd(keys = ("image","label"), spatial_size = dims_tuple), Rand3DElasticd(keys = ("image","label"), sigma_range = (0.5,1), magnitude_range = (0.1, 0.4), prob=0.4, shear_range=(0.1, -0.05, 0.0, 0.0, 0.0, 0.0), scale_range=0.5, padding_mode= "zeros"),
              RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=1),RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=0),RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=2)])
    val_transforms = Compose([SpatialPadd(keys = ("image","label"),spatial_size = dims_tuple)])
    train_patches_dataset = CacheDataset(data=arr_train_dict ,transform = train_transforms, cache_rate =0.25, copy_cache=False, progress=True) # dataset with cache mechanism that can load data and cache deterministic transforms’ result during training.
    validate_patches_dataset = CacheDataset(data=arr_val_dict ,transform = val_transforms, cache_rate = 0.25, copy_cache=False,progress=True)
   
    filter = filter_num 
    cnn_depth = depth 
    layer_per_block = num_conv 
    bottleneck = cnn_depth # set num convs in bottleneck equal to depth
    
    channels = ()
    for i in range(cnn_depth):
        channels += layer_per_block*(filter,)
        filter *=2
    channels+=  bottleneck*(filter,) 

    strides = ()
    for i in range(cnn_depth):
        strides += (2,) + (1,)*(layer_per_block -1)
    strides += (bottleneck-1) * (1,)
    
    CNN = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels, 
        strides=strides,
        dropout=0.2,
        norm='BATCH'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    CNN.apply(monai.networks.normal_init)
    CNN_model = CNN.to(device)
    trainable_params = sum(p.numel() for p in CNN_model.parameters() if p.requires_grad)

    training_steps = int(np_tr / batch_size) # number of training steps per epoch
    validation_steps = int(np_va / batch_size) # number of validation steps per epoch

    train_loader = DataLoader(train_patches_dataset, batch_size=batch_size, shuffle=False, num_workers=32) 
    val_loader = DataLoader(validate_patches_dataset, batch_size=batch_size, shuffle=False,num_workers=32)

    import time
    learning_rate = float(lr)
    betas = (0.5, 0.999)
    cnn_opt = torch.optim.Adam(CNN_model.parameters(), lr = learning_rate, betas=betas)
    patience = 22# epochs it will take for training to terminate if no improvement
    early_stopping = EarlyStopping(patience=patience, verbose=True, path = f'{output_dir}/checkpoint.pt')
    start = time.time() # initializing variable to calculate training time
    max_epochs = 800 # max total iterations over entire training set
    
    #root_mean_squared = MeanSquaredError(squared = False).to(device) #rmse metric calculated at the end of each epoch for training and val
    if loss_func == 'mae':
        loss = torch.nn.L1Loss().to(device)
    elif loss_func == 'ssim':
        loss = SSIM(gaussian_kernel=True, sigma=1.5, reduction='elementwise_mean').to(device)
    elif loss_func == 'both':
        loss1 = torch.nn.L1Loss().to(device)
        loss2 = SSIM(gaussian_kernel=True, sigma=1.5, reduction='elementwise_mean').to(device)
    
    val_losses = [float('inf')] # list of validation loss calculated at the end of each epoch
    train_losses = [float('inf')] # list of training loss calculated at the end of each epoch
    
    #mae_val = [0] # list of validation loss calculated at the end of each epoch
    #epoch_loss_values = [0] # list of training loss calculated at the end of each epoch

    for epoch in range(max_epochs):
        CNN_model.train() # setting model to training mode
        avg_train_loss = 0 # will hold sum of all training losses in epoch and then average
        #epoch_loss = 0 # total traininig loss in an epoch
        progress_bar(
                index=epoch+1, # displays what step we are of current epoch, our epoch number, training  loss
                count = max_epochs, 
                desc= f"epoch {epoch + 1}, training mae loss: {train_losses[-1]:.4f}, validation mae metric: {val_losses[-1]:.4f}",
                newline = True) # progress bar to display current stage in training

        for i,train_batch in enumerate(train_loader): # iterating through dataloader
            gad_images =train_batch["image"].cuda()# batch with gad images
            nongad_images = train_batch["label"].cuda() # batch with nongad images
            #plt.subplot(2, 1, 1)
            #plt.imshow(gad_images[0, 0,: ,:, 20].cpu().data.numpy(), cmap ="gray")
            #plt.subplot(2, 1, 2)
            cnn_opt.zero_grad()
            degad_images = CNN_model(gad_images) # feeding CNN with gad images
            if loss_func == 'mae':
                train_loss = loss(degad_images, nongad_images)
            elif loss_func == 'ssim':
                train_loss = 1- loss(degad_images, nongad_images)
            elif loss_func ==  'both':
                train_loss=  0.5*loss1(degad_images, nongad_images) + 0.5*(1-loss1(degad_images, nongad_images))
            train_loss.backward()
            cnn_opt.step()
            avg_train_loss += train_loss.item() 
        avg_train_loss /= training_steps
        train_losses.append(avg_train_loss) # append total epoch loss divided by the number of training steps in epoch to loss list
        CNN_model.eval() #setting model to evaluation mode for validation
        with torch.no_grad(): #we do not update weights/biases in validation training, only used to assess current state of model
            avg_val_loss = 0 # will hold sum of all validation losses in epoch and then average
            for i,val_batch in enumerate(val_loader): # iterating through dataloader
                gad_images =val_batch["image"].cuda()# batch with gad images
                nongad_images = val_batch["label"].cuda() # batch with nongad images
                degad_images = CNN_model(gad_images)
                if loss_func == 'mae':
                    val_loss = loss(degad_images, nongad_images)
                elif loss_func == 'ssim':
                    val_loss = 1- loss(degad_images, nongad_images)
                elif loss_func ==  'both':
                    val_loss=  0.5*loss1(degad_images, nongad_images) + 0.5*(1-loss1(degad_images, nongad_images))
                avg_val_loss += val_loss 
            avg_val_loss = avg_val_loss.item()/validation_steps #producing average val loss for this epoch
            val_losses.append(avg_val_loss) 
            early_stopping(avg_val_loss, CNN_model) # early stopping keeps track of last best model

        if early_stopping.early_stop: # stops early if validation loss has not improved for {patience} number of epochs
            print("Early stopping") 
            break


    end = time.time()
    time = end - start
    print(time)

    with open (f'{output_dir}/model_stats.txt', 'w') as file:  
        file.write(f'Training time: {time}\n') 
        file.write(f'Number of trainable parameters: {trainable_params}\n')
        file.write(f'Training loss: {train_losses[-patience]} \nValidation loss: {early_stopping.val_loss_min}')

    plt.figure(figsize=(12,5))
    plt.plot(list(range(len(train_losses))), train_losses, label="Training Loss")
    plt.plot(list(range(len(val_losses))),val_losses , label="Validation Loss")
    plt.grid(True, "both", "both")
    plt.legend()
    plt.savefig(f'{output_dir}/lossfunction.png')


    CNN_model.load_state_dict(torch.load(f'{output_dir}/checkpoint.pt'))
    CNN_model.eval()

    # running inference 
    test_gad_t1= sorted(glob('/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/sub-P*/*rescaled_T1w.nii.gz'))[-1:] #TODO: MOVE PASSINGDATASET INSIDE SNAKEMAKE
    # gad images who's corresponding nongad images underwent a rigid transform
    test_nongad_t1= sorted(glob('/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/sub-P*/*nongad_normalized_fcm.nii.gz'))[-1:] # nongad images which underwent a rigid transform and underwent fcm normalization
    test_files = [{"image": gad_name, "label": nongad_name} for gad_name, nongad_name in zip(test_gad_t1,test_nongad_t1)] #creates list of dictionaries, with gad and nongad images labelled


    inference_transforms = Compose( #loading full image
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"])])


    infer_ds = Dataset(data=test_files, transform=inference_transforms) 
    infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=True) #using pytorch's dataloader



    # In[ ]:


    degad_imgs = []
    gad_infer_imgs = []
    nongad_infer_imgs = []
    for infer_imgs in infer_loader:
        gad_infer_imgs.append(infer_imgs["image"])
        nongad_infer_imgs.append(infer_imgs["label"])
        output_degad_img = sliding_window_inference(inputs = infer_imgs["image"].to('cpu'), roi_size = dims_tuple, sw_batch_size= 5, predictor = CNN_model.to('cpu'), overlap = 0.25, mode = "gaussian", sw_device= 'cpu', device = 'cpu', progress = True )
        degad_imgs.append(output_degad_img)


    # In[ ]:


    for i in range(len(degad_imgs)): #looping thru number of output files
        degad_img =degad_imgs[i][0][0] # reshaping to exclude batch and channels (only one channel)
        gad_image= nibabel.load(test_files[i]["image"]) # getting original gad image back to compare to 
        gad_image_file = test_files[i]["image"]
        print(gad_image_file)
        sub = os.path.basename(gad_image_file).split("_")[0]
        degad_name = f'{sub}_acq-degad_T1w.nii.gz'
        degad_file = nibabel.Nifti1Image(degad_img.detach().numpy()*100, affine= gad_image.affine,header= gad_image.header) # with same header as inference gad 
        output_dir_test = f'{output_dir}/test'
        os.makedirs(f'{output_dir_test}/bids/{sub}/ses-pre/anat', exist_ok=True)# save in bids format
        output_path = f'{output_dir_test}/bids/{sub}/ses-pre/anat/{degad_name}'
        nibabel.save(degad_file,output_path) 


    # In[ ]:


    import random
    ## generating random whole brain slices
    fig, axes = plt.subplots(8, 4,figsize=(10,25))
    plt.suptitle('Whole brain slices: Gad, NonGad, Degad,Subtraction map')

    for i in range (1,33,4):
        plt.subplot(8, 4, i)
        x = random.randint(80, 190)
        plt.imshow(gad_infer_imgs[0][0, 0,40:210 ,40:150, x].cpu().data.numpy(), cmap ="gray")

        plt.subplot(8, 4, i+1)
        plt.imshow(nongad_infer_imgs[0][0, 0, 40:210 , 40:150, x].cpu().data.numpy(), "gray")

        plt.subplot(8, 4, i+2)
        plt.imshow(degad_imgs[0][0, 0, 40:210,40:150, x].cpu().data.numpy(), "gray")
    
        plt.subplot(8, 4, i+3)
        noise_vector = degad_imgs[0][0,0,:,:,:] - nongad_infer_imgs[0][0,0,:,:,:] 
        #pos values are where model overestimated intensities and neg values are where the model underestimated
        plt.imshow(noise_vector[40:210,40:150, x].cpu().data.numpy(), "seismic",vmin=-1,vmax=1)
        plt.colorbar()
    plt.savefig(f'{output_dir}/test/figure_whole_brain.png')


    # In[ ]:


    #generating random 32x32 slices
    fig, axes = plt.subplots(8, 4,figsize=(10,20))
    plt.suptitle('Patches: Gad, NonGad, Degad,subtraction map')

    for i in range (1,33,4):

        x = random.randint(40, 190)
        y = random.randint(40, 190)
        z = random.randint(40, 190)
        plt.subplot(8, 4, i)
        plt.imshow(gad_infer_imgs[0][0, 0, x:x+32,y:y+32 ,50].cpu().data.numpy(), cmap ="gray")
        plt.subplot(8, 4, i+1)
        plt.imshow(nongad_infer_imgs[0][0, 0, x:x+32,y:y+32 ,50].cpu().data.numpy(), "gray")
        plt.subplot(8, 4, i+2)
        plt.imshow(degad_imgs[0][0, 0, x:x+32,y:y+32,50].cpu().data.numpy(), "gray")
        plt.subplot(8, 4, i+3)
        noise_vector = degad_imgs[0][0,0,:,:,:] - nongad_infer_imgs[0][0,0,:,:,:] 
        #pos values are where model overestimated intensities and neg values are where the model underestimated
        plt.imshow(noise_vector[x:x+32,y:y+32,50].cpu().data.numpy(), "seismic",vmin=-1,vmax=1)
        plt.colorbar()

    plt.savefig(f'{output_dir}/test/figure_32_patches.png')  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN degad model with specified parameters.")
    parser.add_argument("--input", nargs='+', required=True, help="Path to the training and validation data, in that order")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size for training.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("--ini_filter", type=int, required=True, help="Number of filters in initial layer.")
    parser.add_argument("--depth", type=int, required=True, help="Depth of U-Net.")
    parser.add_argument("--num_conv", type=int, required=True, help="Number of convolutions in each layer.")
    parser.add_argument("--loss", required=True, help="Type of loss function to apply: mae, ssim or both.")

    args = parser.parse_args()
    input_files = args.input
    patch_size = args.patch_size
    batch_size = args.batch_size
    lr = args.lr
    filter_num=args.ini_filter
    depth= args.depth
    num_conv=args.num_conv
    loss=args.loss

    train_model(input_files,patch_size, batch_size,lr,filter_num,depth,num_conv, loss)




