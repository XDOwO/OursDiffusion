#train.py
from dataset import DiffusionDataset
from diffusion import Diffusion
from unet import Unet, device
from torch.utils.data import DataLoader
import torch
dataset = DiffusionDataset("./dataset")
print("dataset length:",len(dataset))
dataloader=DataLoader(dataset,batch_size=16,shuffle = True)
unet = Unet(in_ch=3,model_ch=64,out_ch=3,num_res=4,time_emb_mult=2,dropout=0,ch_mult = (1,1,2,2,4,4), num_heads=1, attention_at_height = 3).to(device)
unet.load_state_dict(torch.load("/home/fish/OursDiffusion/save/model/OurDiffusion_36.pth"))
diffusion_model = Diffusion(4000,unet,"./save","cosine")

if __name__ == "__main__":
    diffusion_model.train_model(10000,dataloader,0.00005,save_interval = 1)
