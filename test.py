#train.py
from dataset import DiffusionDataset
from diffusion import Diffusion
from unet import Unet, device
from torch.utils.data import DataLoader
import torch
import os
from milesAE.aeFunc import Encode_tensor
from torchvision import transforms


save_path = "./save/plt_test"
dataset = DiffusionDataset("./dataset")
print("dataset length:",len(dataset))
dataloader=DataLoader(dataset,1,shuffle = True)
unet = Unet(in_ch=3,model_ch=64,out_ch=3,num_res=4,time_emb_mult=2,dropout=0,ch_mult = (1,1,2,2,4,4), num_heads=1, attention_at_height = 3).to(device)
unet.load_state_dict(torch.load("/home/fish/OursDiffusion/save/model/OurDiffusion_47.pth"))
unet.eval()
diffusion_model = Diffusion(4000,unet,save_path,"cosine")
if __name__ == "__main__":
    if not os.path.exists(save_path):
      os.makedirs(os.path.join(save_path,"plt"))
    gt, emp, xy, car, xylen = next(iter(dataloader))
    emp=emp[0].unsqueeze(0).to(device)
    car=car[0].unsqueeze(0).to(device)
    xy=xy[0].unsqueeze(0).to(device)
    gt=gt[0].unsqueeze(0).to(device)
    xylen=xylen[0].unsqueeze(0).to(device)
    noise = torch.randn_like(gt[0])

    mask = torch.full_like(gt[0],-1)
    for xy_n in range(xy.shape[1]):
      x,y = xy[0][xy_n]
      x_rad, y_rad = xylen[0][xy_n]
      x_rad //= 2
      y_rad //= 2

      if x==y==0:
        continue
      mask[:,max(x-x_rad,0):min(x+x_rad,256):,max(0,y-y_rad):min(y+y_rad,256)] = 1
    ans = gt[0]
    ans[mask == -1] = -1
    t2 =transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.ToPILImage(),
    ])
    t2(ans).save("tesst.png")

    noised_x,noise = diffusion_model.show_forward_process_img(gt[0])
    noised_x = noised_x.unsqueeze(0).to(device)
    noise = noise.unsqueeze(0).to(device)
    emp=Encode_tensor(diffusion_model.show_backward_process_img(gt.shape,emp,car,xy).unsqueeze(0))

    ans = diffusion_model.show_backward_process_img(gt.shape,emp,car,xy)
    mask = torch.full_like(ans,-1)
    for xy_n in range(xy.shape[1]):
      x,y = xy[0][xy_n]
      x_rad, y_rad = xylen[0][xy_n]
      x_rad //= 2
      y_rad //= 2

      if x==y==0:
        continue
      mask[:,max(x-x_rad,0):min(x+x_rad,256):,max(0,y-y_rad):min(y+y_rad,256)] = 1
    ans[mask == -1] = -1
    t2 =transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.ToPILImage(),
    ])
    t2(ans).save("test.png")

    
