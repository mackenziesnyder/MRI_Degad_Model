import config
from torch.utils.datainport Dataset
import numpy as np 
import os
from Pil import Image

class NogadGadDataset(Dataset):
    def __init__(self, root_nogad, root_gad, transform=None):
        self.root_nogad = root_nogad
        self.root_gad = root_gad
        self.transform = transform

        self.nogad_imgs = os.listdir(root_nogad)
        self.gad_imgs = os.listdir(root_gad)
        self.length_dataset = max(len(self.nogad_imgs, self.gad_imgs))
        self.length_nogad_imgs = len(self.nogad_imgs)
        self.length_gad_imgs = len(self.gad_imgs)
    
    def __len__(self):
        return self.length_dataset
    
    def __get_item__(self, index):
        nogad_img = self.nogad_imgs[index % self.length_nogad_imgs]
        gad_img = self.gad_imgs[index % self.length_gad_imgs]

        nogad_path = os.path.join(self.root_nogad, nogad_img)
        gad_path = os.path.join(self.root_gad, gad_img)

        nogad_img = np.array(Image.open(nogad_path))
        gad_img = np.array(Image.open(gad_path))

        if self.transform:
            augmentations = self.transform(image=nogad_img, image0=gad_img)
            nogad_img = augmentations["image"]
            gad_img = augmentations["image0"]
        
        return ngad_img, gad_img 

