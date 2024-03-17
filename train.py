#train.py
from dataset import DiffusionDataset
from diffusion import Diffusion
from unet import Unet, device
from torch.utils.data import DataLoader
import torch
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',"--dataset_path",default="./dataset",help="Path to dataset")
    parser.add_argument('-s',"--save_path",default='./save',help="Path to store models and test pictures")
    parser.add_argument("-r","--ratio",type=int,default=1,help="Loss ratio of road and cars, if ratio > 1, then model will tend to generate road more.")
    parser.add_argument("-d","--device",type=int,default=0,help="Index of GPU")

    args = parser.parse_args()

    save_path = args.save_path
    ratio = args.ratio
    device = "cuda:"+str(args.device)

    dataset = DiffusionDataset("./dataset")
    print("dataset length:",len(dataset))
    print("GPU in use:",device)
    dataloader=DataLoader(dataset,batch_size=20,shuffle = True)
    unet = Unet(in_ch=3,model_ch=64,out_ch=3,num_res=4,time_emb_mult=2,dropout=0,ch_mult = (1,1,2,2,4,4), num_heads=1, attention_at_height = 3).to(device)
    unet.load_state_dict(torch.load("/home/fish/OursDiffusion/save/model/OurDiffusion_20.pth"))
    diffusion_model = Diffusion(4000,unet,"./save","cosine",device = device)
    diffusion_model.train_model(10000,dataloader,lr = 0.00002,ratio = ratio,save_interval = 1)
