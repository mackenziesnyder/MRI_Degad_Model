import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/UWO/msnyde26/graham/scratch/cyclegan_data/"
AL_DIR = "/home/UWO/msnyde26/graham/scratch/cyclegan_data/"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_DISC_H = "critich.pth.tar"
CHECKPOINT_SISC_Z = "criticz.pth.tar"

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

transform_pipeline = transforms.Compose([
    PadToSize((256, 256)),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
