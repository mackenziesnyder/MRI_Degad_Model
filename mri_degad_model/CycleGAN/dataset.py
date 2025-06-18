import config
from torch.utils.data import Dataset
import os
from PIL import Image

class NogadGadDataset(Dataset):
    def __init__(self, root_nogad, root_gad, transform=None):
        self.root_nogad = root_nogad
        self.root_gad = root_gad
        self.transform = transform

        self.nogad_imgs = os.listdir(root_nogad)
        self.gad_imgs = os.listdir(root_gad)
        self.length_dataset = max(len(self.nogad_imgs), len(self.gad_imgs))
        self.length_nogad_imgs = len(self.nogad_imgs)
        self.length_gad_imgs = len(self.gad_imgs)
    
    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        nogad_img_name = self.nogad_imgs[index % self.length_nogad_imgs]
        gad_img_name = self.gad_imgs[index % self.length_gad_imgs]

        nogad_path = os.path.join(self.root_nogad, nogad_img_name)
        gad_path = os.path.join(self.root_gad, gad_img_name)

        nogad_img = Image.open(nogad_path).convert("L")  # grayscale PIL image
        gad_img = Image.open(gad_path).convert("L")

        if self.transform:
            nogad_img = self.transform(nogad_img)
            gad_img = self.transform(gad_img)

        return nogad_img, gad_img