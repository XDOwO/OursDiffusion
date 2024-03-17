#diffusion.py
import torch
import math
import os
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm.auto import tqdm
from random import randint


class Diffusion:
  def __init__(self,T,model,save_path,schedule_type="linear",device="cuda:0",scale=None):
    self.device = device
    # self.device = "cpu"
    self.T = T
    if schedule_type == "linear":
      self.betas = self.linear_schedule(T).to(self.device)
    elif schedule_type == "cosine":
      self.betas = self.cosine_schedule(T).to(self.device)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas,dim = 0).to(self.device)
    self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1])).to(self.device)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
    self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
    self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(self.device)
    self.posterior_log_variance_clipped = torch.log(torch.cat((self.posterior_variance[1:2], self.posterior_variance[1:]))).to(self.device)
    self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(self.device)
    self.posterior_mean_coef2 = (
        (1.0 - self.alphas_cumprod_prev)
        * torch.sqrt(self.alphas)
        / (1.0 - self.alphas_cumprod)
    ).to(self.device)
    self.save_path = save_path
    self.model = model.to(self.device)
  def linear_schedule(self,T,scale=None,start=0.0001,end=0.02):
    '''The schedule is from improved diffusion'''
    if scale == None:
      scale = 1000/T
    return torch.linspace(start*scale,end*scale,T,dtype=torch.float32)
  def cosine_schedule(self,T,s=0.008,max_beta=0.9999):
    '''The schedule is from improved diffusion'''
    def alpha_bar(t):
      return math.cos((t+0.008) / 1.008 * math.pi /2) ** 2
    betas = []
    for i in range(T):
      t1 = i / T
      t2 = (i+1) / T
      betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas,dtype=torch.float32)

  def forward_process(self, x_0, t, noise=None):
    '''
    Given x_0,t , get x_t
    '''
    if noise == None:
      noise = torch.randn_like(x_0).to(self.device)
    assert noise.shape == x_0.shape, "Noise shape{} should be the same as x shape{}".format(noise.shape,x_0.shape)
    # print(t.shape,x_0.shape,noise.shape,self.sqrt_alphas_cumprod.shape,self.sqrt_one_minus_alphas_cumprod.shape,(x_0*self.sqrt_alphas_cumprod[t]).shape,(noise*self.sqrt_one_minus_alphas_cumprod[t]).shape)
    return x_0*extract(self.sqrt_alphas_cumprod,t,x_0.shape) + noise*extract(self.sqrt_one_minus_alphas_cumprod,t,x_0.shape),noise

  def show_forward_process_img(self,x,noise = None):
    plt.figure(figsize=(1024/72, 576/144))
    plt.subplot(1,11,1)
    t = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.ToPILImage(),
    ])
    plt.imshow(t(x))
    if noise == None:
      noise = torch.randn_like(x)
    for i in range(2,12):
      plt.subplot(1,11,i)
      noised_x,noise = self.forward_process(x,t = torch.full((x.shape[0],),self.T/10*(i-1)-1,dtype=torch.long).to(self.device),noise=noise)
      plt.imshow(t(noised_x))
      plt.axis("off")
    
    plt.savefig(os.path.join(self.save_path,"plt","forward.png"))
    plt.close()

    return noised_x,noise
  @torch.no_grad
  def backward_step(self,x_t,t,rds = None,cars = None,xys = None,eps = None):
    '''
    Given x_t,t,and else, get x_(t-1)
    '''
    
    noise = torch.randn_like(x_t)
    eps = eps if eps is not None else self.model(x_t,t,rds,cars,xys)
    pred_xstart = extract(self.sqrt_recip_alphas,t,x_t.shape)*x_t - extract(self.sqrt_recipm1_alphas_cumprod,t,x_t.shape)*eps
    mean = extract(self.posterior_mean_coef1, t, x_t.shape) * pred_xstart + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    log_variance = extract(torch.log(torch.cat((self.posterior_variance[1:2], self.betas[1:]),dim=0)), t, x_t.shape)    
    nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
    return mean + nonzero_mask * torch.exp(0.5*log_variance)*noise
    # variance = self.posterior_variance[t]
    # betas_t = self.betas[t]
    # sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
    # sqrt_recip_alphas_t =self.sqrt_recip_alphas[t]
    # mean = self.model(x_t,t,rds,cars,xys) if n2 == None else n2
    # mean = sqrt_recip_alphas_t*(x_t-betas_t * mean / sqrt_one_minus_alphas_cumprod_t)
    # if t==0:
    #   return mean
    # else:
    #   return mean+torch.sqrt(variance)*noise
  
  @torch.no_grad
  def backward_process(self,shape,rds,cars,xys,noise=None,save_intermidate = False):
    if noise == None:
      noise = torch.randn(*shape).to(self.device)
      sample = noise
    else:
      sample = noise
    intermidate = []
    for i in tqdm(range(0,self.T)[::-1],desc="Backward processing", position=0, leave=True):
      if save_intermidate:
        intermidate.append(sample[0] if len(sample.shape)==4 else sample)
      t = torch.tensor([i]*shape[0]).to(self.device)
      sample = self.backward_step(sample,t,rds,cars,xys)
    intermidate.append(sample[0] if len(sample.shape)==4 else sample)
    return sample if not save_intermidate else intermidate

  @torch.no_grad
  def show_backward_process_img(self,shape,rds,cars,xys,noise=None):
    lis = self.backward_process(shape,rds,cars,xys,noise,True)
    plt.figure(figsize=(1024/72, 576/144))
    t = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Resize((64,64),antialias=True),
        transforms.ToPILImage(),
    ])
    t2 =transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.ToPILImage(),
    ])
    plt.subplot(1,11,11)
    plt.imshow(t(lis[0]))
    plt.axis("off")
    for i in range(1,11):
      plt.subplot(1,11,i)
      plt.imshow(t(lis[int(self.T-self.T/10*(i-1)-1)]))
      plt.axis("off")

    plt.savefig(os.path.join(self.save_path,"plt","backward.png"))
    t2(lis[-1]).save(os.path.join(self.save_path,"test_res","res.png"))
    plt.close()
    return lis[-1]

  def weighted_mse(self,targets,predicts,xys,xylens,full_weight=1000,mode="road"):
    assert targets.shape[0]==xys.shape[0], "target and xy batch should be the same."
    mask = torch.full_like(targets,1) if mode == "road" else torch.full_like(targets, -1)
    w,h = targets.shape[2:]
    for b in range(targets.shape[0]):
      for xy_n in range(xys.shape[1]):
        x,y = xys[b][xy_n]
        x_rad, y_rad = xylens[b][xy_n]
        x_rad //= 2
        y_rad //= 2

        if x==y==0:
          continue
        mask[b][:,max(0,y-y_rad):min(y+y_rad,h):,max(x-x_rad,0):min(x+x_rad,w)] = -1 if mode == "road" else 1
    
    # t2 =transforms.Compose([
    #     transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.ToPILImage(),
    # ])
    targets[mask == -1] = -1
    predicts[mask == -1] = -1
    # t2(targets[0]).save("/home/fish/OursDiffusion/testimg/target.png")
    # t2(mask[0]).save("/home/fish/OursDiffusion/testimg/mask.png")
    # assert 1==0
        
    # xt,_ = self.forward_process(gt,ts,targets)
    # xt_minus_1,_ = self.forward_process(gt,torch.clamp(ts,min=0),noise=targets)
    # predicts = self.backward_step(xt,ts,eps = predicts)
    return torch.mean(((targets-predicts)**2)*full_weight)
  def train_model(self,epoch,dataloader,lr,ratio=1,save_interval = 20):
    import torch.optim as optim
    import os.path
    if not os.path.exists(self.save_path):
      os.makedirs(os.path.join(self.save_path,"plt"))
      os.makedirs(os.path.join(self.save_path,"model"))
      os.makedirs(os.path.join(self.save_path,"test_res"))

    loss = nn.MSELoss()
    opt = optim.Adam(self.model.parameters(),lr = lr)
    noise = None
    for E in range(epoch):
      all_loss_r = 0
      all_loss_c = 0
      loop = tqdm(dataloader,desc="Training", position=0, leave=True)
      for gt, emp, xy, car, xylen in loop:
        # t = torch.randint(0,self.T,size=(gt.shape[0],),dtype=torch.long)
        gt, emp, xy, car, xylen = gt.to(self.device), emp.to(self.device), xy.to(self.device), car.to(self.device), xylen.to(self.device)
        t = torch.randint(low=0, high=self.T, size=(gt.shape[0],), dtype=torch.long).to(self.device)
        noise = torch.randn_like(gt).to(self.device)
        x_t,_ = self.forward_process(gt,t,noise)
        
        # Calculate car loss first
        output = self.model(x_t,t,emp,car,xy)
        loss_c = self.weighted_mse(output,noise,xy,xylen,1,"car")
        loss_c_cpu = loss_c.cpu().item()
        # loss_c.backward()
        # opt.step()
        all_loss_c+=loss_c_cpu
        # opt.zero_grad()

        # Calculate road loss second
        # output = self.model(x_t,t,emp,car,xy)
        loss_r = self.weighted_mse(output,noise,xy,xylen,10,"road")
        loss_r_cpu = loss_r.cpu().item()
        loss_sum = loss_r+loss_c
        loss_sum.backward()
        # loss_r.backward()
        opt.step()
        all_loss_r+=loss_r_cpu
        opt.zero_grad()
        loop.set_description(f'Epoch [{E+1}/{epoch}]')
        loop.set_postfix(loss=loss_r_cpu+loss_c_cpu)
      print("Avg loss_road at epoch {}:".format(E),all_loss_r/len(dataloader))
      print("Avg loss_car at epoch {}:".format(E),all_loss_c/len(dataloader))
      if E % save_interval == 0:
        torch.save(self.model.state_dict(),os.path.join(self.save_path,"model","OurDiffusion_{}.pth".format(E//save_interval)))
        print("Model saved!")
        emp=emp[0].unsqueeze(0)
        car=car[0].unsqueeze(0)
        xy=xy[0].unsqueeze(0)
        gt=gt[0].unsqueeze(0)
        self.show_backward_process_img(gt.shape,emp,car,xy)
        os.rename(os.path.join(self.save_path,"plt","backward.png"),os.path.join(self.save_path,"plt","backward_{}.png".format(E//save_interval)))
        os.rename(os.path.join(self.save_path,"test_res","res.png"),os.path.join(self.save_path,"test_res","res_{}.png".format(E//save_interval)))

def extract(a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        # print(t)
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

