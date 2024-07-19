import torch
from models.unet import UNet

from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class OCTADataset(Dataset):

    def __init__(self, path, is_train=True):
        self.path = path
        self.mask_img_paths = sorted(glob.glob(path+"mask/*"))
        self.tampered_img_paths = sorted(glob.glob(path+"tampered/*"))
        if is_train:
            self.perfect_img_paths = sorted(glob.glob(path+"perfect/*"))
        self.is_train = is_train

    def __len__(self,):
        return len(self.mask_img_paths)
    
    def __getitem__(self, idx):

        mask = self.mask_img_paths[idx]
        mask = transforms.ToTensor()(Image.open(mask).convert('1'))

        tamper = self.tampered_img_paths[idx]
        tamper = transforms.ToTensor()(Image.open(tamper).convert('RGB'))

        train_img = torch.concat((tamper, mask), dim=0)

        return_dict = {
                            "mask": mask,
                            "tamper": tamper,
                            "img_set": train_img,
                            "moh": train_img
                      }
        if self.is_train:
            perfect = self.perfect_img_paths[idx]
            perfect = transforms.ToTensor()(Image.open(perfect).convert('RGB'))
            return_dict["perfect"] = perfect
            
        
        return return_dict



model = UNet(n_channels=4, n_classes=3).to(device)

try:
    model.load_state_dict(torch.load("epoch-last.pth")['model']['sd'])
except:
    print("epoch-last.pth NOT FOUND! Using UNet_epoch-last.pth")
    model.load_state_dict(torch.load("UNet_epoch-last.pth")['model']['sd'])


BS = 1
dataset = OCTADataset(path="test/", is_train=False)
test_dataloader = DataLoader(dataset,
                    batch_size=BS,
                    shuffle=True,
                    num_workers=8, pin_memory=True)


model.eval()

for idx, batch in enumerate(tqdm(test_dataloader, leave=False)):
    for k, v in batch.items():
            batch[k] = v.cuda()
    pred = model(batch["img_set"]).clamp(0, 1)

    transforms.ToPILImage()(pred[0].to("cpu")).save("test_"+str(idx)+".jpg")
