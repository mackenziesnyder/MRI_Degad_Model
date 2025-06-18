import nibabel as nib
from sklearn.metrics import  mean_absolute_error
from skimage.metrics import structural_similarity
import numpy as np

# compute the mean absolute error between 2 nii files
def compute_mae(degad, nogad):
    
    # flatten arrays for mean_absolute_error function
    mae = mean_absolute_error(degad.flatten(), nogad.flatten())
    
    return mae

def compute_ssim(degad, nogad):

    # Normalize images to 0-1, avoid dividing by 0
    degad = (degad - np.min(degad)) / (np.max(degad) - np.min(degad) + 1e-8)
    nogad = (nogad - np.min(nogad)) / (np.max(nogad) - np.min(nogad) + 1e-8)

    # Compute SSIM per slice along the axial direction
    ssim_scores = []
    for i in range(degad.shape[2]):
        score = structural_similarity(degad[:, :, i], nogad[:, :, i], data_range=1.0)
        ssim_scores.append(score)

    return np.mean(ssim_scores)


