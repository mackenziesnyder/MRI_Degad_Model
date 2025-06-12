import nibabel as nib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity
from skimage.transform import resize
import numpy as np

def reshape_d(degad, nogad):
    if degad.shape != nogad.shape:
        print("Warning: shape mismatch, resizing degad to match nogad")
        degad = resize(degad, nogad.shape, preserve_range=True, anti_aliasing=True)
    return degad

# compute the mean squared error between 2 nii files
def compute_mse(degad, nogad):

    degad = reshape_d(degad, nogad)
    # flatten arrays for mean_squared_error function
    mse = mean_squared_error(degad.flatten(), nogad.flatten())
    
    return mse

# compute the mean absolute error between 2 nii files
def compute_mae(degad, nogad):

    degad = reshape_d(degad, nogad)
    # flatten arrays for mean_absolute_error function
    mae = mean_absolute_error(degad.flatten(), nogad.flatten())
    
    return mae

# Function to threshold values based on binary range [0, 1]
def threshold_values(arr):
    arr[arr > 1] = 1
    arr[arr < 0] = 0

def compute_dice(degad, nogad):

    degad = reshape_d(degad, nogad)
    # Apply thresholding to keep values within binary range
    degad_t = threshold_values(degad)
    nogad_t = threshold_values(nogad)

    # Calculate the area of overlap
    area_overlap = np.sum(degad_t * nogad_t)

    # Calculate the total sum
    total_sum = np.sum(degad_t) + np.sum(nogad_t)

    # Calculate Dice coefficient
    dice_pred = (2 * area_overlap) / total_sum

    return dice_pred

def compute_ssim(degad, nogad):
    degad = reshape_d(degad, nogad)

    # Normalize images to 0-1, avoid dividing by 0
    degad = (degad - np.min(degad)) / (np.max(degad) - np.min(degad) + 1e-8)
    nogad = (nogad - np.min(nogad)) / (np.max(nogad) - np.min(nogad) + 1e-8)

    # Compute SSIM per slice along the axial direction
    ssim_scores = []
    for i in range(degad.shape[2]):
        score = structural_similarity(degad[:, :, i], nogad[:, :, i], data_range=1.0)
        ssim_scores.append(score)

    return np.mean(ssim_scores)


