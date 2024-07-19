from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam
import glob
from PIL import Image
from models.unet import UNet
from torch.utils.data import DataLoader
import torch
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


BS = 64

dataset = OCTADataset(path="data/train/")


train_dataloader = DataLoader(dataset,
                    batch_size=BS,
                    shuffle=True,
                    num_workers=8, pin_memory=True)

dataset = OCTADataset(path="test/", is_train=False)
test_dataloader = DataLoader(dataset,
                    batch_size=BS,
                    shuffle=True,
                    num_workers=8, pin_memory=True)

model = UNet(n_channels=4, n_classes=3).to(device)


optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()


model_spec = {}
optimizer_spec = {}


for epoch in range(10000):
    tot = 0

    model.train()
    for batch in tqdm(train_dataloader, leave=False, desc="Epoch: "+ str(epoch)):

        for k, v in batch.items():
                batch[k] = v.cuda()


        pred = model(batch["img_set"])
        loss1 = loss_fn(pred, batch["perfect"])
        loss2 = loss_fn(pred*batch["mask"], batch["perfect"]*batch["mask"])
        loss= loss1 + loss2
        tot+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:", epoch, "  Average Loss:", tot/len(train_dataloader))

    # model_spec = config['model']
    
    model_spec['sd'] = model.state_dict()
    # optimizer_spec = config['optimizer']
    optimizer_spec['sd'] = optimizer.state_dict()
    sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'epoch': epoch
                }
    torch.save(sv_file, 'epoch-last.pth')


    model.eval()

    ################## UNET ###################
    for batch in tqdm(test_dataloader, leave=False, desc="Epoch: "+ str(epoch)):
        for k, v in batch.items():
                batch[k] = v.cuda()
        pred = model(batch["img_set"]).clamp(0, 1)

    transforms.ToPILImage()(pred[0].to("cpu")).save("check.jpg")

    ################## UNET ###################


