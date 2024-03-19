#unet.py
import torch.nn as nn
import torch
import math

class MultipleSequential(nn.Sequential):
  def forward(self,x,emb,*unused):
    for layer in self:
      x = layer(x,emb)
    return x

class T_R_C_X_Embedding(nn.Module):
  def __init__(self, xy_nums=32, xy_max_size=256, input_dim=1024,output_dim=64):
    assert output_dim >= xy_nums, "output_dim({}) >= xy_nums({}) not achieved".format(output_dim,xy_nums)
    assert output_dim % xy_nums==0, "output_dim({}) have to be divisible by xy_nums({})".format(output_dim,xy_nums)
    super(T_R_C_X_Embedding, self).__init__()
    self.embedding = nn.Embedding(xy_max_size, output_dim//xy_nums)
    self.carnn = nn.Linear(input_dim,output_dim)
    self.rdnn = nn.Linear(input_dim,output_dim)
    self.nn = nn.Linear(output_dim*4,output_dim)
    self.output_dim = output_dim
    self.norm = nn.GroupNorm(num_groups=32, num_channels=1024)
    self.norm2 = nn.GroupNorm(num_groups=32, num_channels=output_dim*4)

  def forward(self, timesteps,rds,cars,xys, max_period=10000):
    '''
    Create embedding from timestep,empty road,car and cars' coordination(xy)

    :param timesteps: 1-D tensor. The shape should be the same as batch size.
    :param rds: 2-D tensor. The shape should be (batch,1024)
    :param cars: 2-D tensor. The shape should be (batch,1024)
    :param xys: 2/3-D tensor. The shape should be (batch,32) or (batch,16,2)
    '''
    
    B,D = cars.shape[:2]
    assert cars.shape[1] == rds.shape[1], "rds.shape[1]({}) shoudl be the same as cars.shape[1]({})".format(rds.shape[1],cars.shape[1])
    cars = self.carnn(self.norm(cars))
    rds = self.rdnn(self.norm(rds))
    #xy
    if len(xys.shape) == 3:
      B,N,C = xys.shape
      xys = xys.view(B,N*C)
    xys = (xys * 255) //639
    emb_xys = self.embedding(xys).view(B, -1)

    #timesteps
    #Credits to improved diffusion
    half = self.output_dim // 2
    freqs = torch.exp(
      -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    emb_time = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if self.output_dim % 2:
      emb_time = torch.cat([emb_time, torch.zeros_like(emb_time[:, :1])], dim=-1)
    x = torch.cat((emb_time,rds,cars,emb_xys),dim=1)
    x = self.nn(self.norm2(x))

    return x

class ResBlock(nn.Module):
  def __init__(self,in_ch,emb_ch,out_ch,dropout=0.2):
    super().__init__()
    self.input_layer = nn.Sequential(
        nn.GroupNorm(num_groups=min(in_ch,32), num_channels=in_ch),
        nn.SiLU(),
        nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1))
    self.emb_layer = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_ch,out_ch),
    )
    self.output_layer = nn.Sequential(
        nn.GroupNorm(num_groups=min(out_ch,32), num_channels=out_ch),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),
    )
    self.skip_connection = nn.Conv2d(in_ch,out_ch,kernel_size=1)
  def forward(self,x,emb):
    xx = self.input_layer(x)
    emb = self.emb_layer(emb)
    while len(emb.shape) < len(xx.shape):
      emb=emb[...,None]
    xx = xx + emb
    xx = self.output_layer(xx)
    return self.skip_connection(x) + xx

class AttentionBlock(nn.Module):
  def __init__(self,ch,num_heads):
    self.ch = ch
    super().__init__()
    self.norm = nn.GroupNorm(num_groups=min(ch,32), num_channels=ch)
    self.qkv = nn.Conv1d(ch,ch*3,1)
    self.attention = nn.MultiheadAttention(embed_dim=ch,num_heads=num_heads,dropout=0.2,batch_first=True)
    self.out = nn.Conv2d(ch,ch,1)
  def forward(self,x,*unused):
    B,C,H,W = x.shape
    x = x.reshape(B,C,-1)
    x = self.norm(x)
    qkv = self.qkv(x)
    qkv = qkv.permute(0,2,1)
    q,k,v = torch.split(qkv,self.ch,2)
    x,_ = self.attention(q,k,v)
    x = x.permute(0,2,1)
    x = x.view(B,C,H,W)
    return self.out(x)

class Downsample(nn.Module):
  def __init__(self,ch,layer_type="conv"):
    super().__init__()
    if layer_type == "conv":
      self.layer = nn.Conv2d(ch,ch,kernel_size=3,stride=2,padding=1)
    elif layer_type == "maxpool":
      self.layer = nn.MaxPool2d(kernel_size=2,stride=2)
    elif layer_type == "avgpool":
      self.layer = nn.AvgPool2d(kernel_size=2,stride=2)
  def forward(self,x,*unused):
    return self.layer(x)

class Upsample(nn.Module):
  def __init__(self,ch,layer_type="interpolate",add_conv = False):
    super().__init__()
    if layer_type == "interpolate":
      self.layer = nn.Upsample(scale_factor=2, mode='bilinear')
    elif layer_type == "convtranspose":
      self.layer = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2)

    if add_conv:
      self.layer2 = nn.Conv2d(ch,ch,kernel_size=3,padding=1)
    else:
      self.layer2 = nn.Identity()
  def forward(self,x,*unused):
    return self.layer2(self.layer(x))

class Unet(nn.Module):
  def __init__(self,in_ch,model_ch,out_ch,num_res,time_emb_mult=1,dropout=0,ch_mult = (1,2,4,8), num_heads=1, attention_at_height = 3):
    super().__init__()
    self.in_ch = in_ch
    self.model_ch = model_ch
    self.out_ch = out_ch
    self.num_res = num_res
    self.dropout = dropout
    self.ch_mult = ch_mult
    self.num_heads = num_heads
    self.emb_ch = model_ch * time_emb_mult
    self.trcx_emb = T_R_C_X_Embedding(xy_nums=32,output_dim=self.model_ch)

    self.emb_block = nn.Sequential(nn.Linear(model_ch,self.emb_ch),
                                   nn.SiLU(),
                                   nn.Linear(self.emb_ch,self.emb_ch)
                                  )

    self.down_blocks = nn.ModuleList()
    self.inn = nn.Conv2d(in_ch,model_ch,kernel_size = 3, padding = 1)
    ch = model_ch
    for height, mult in enumerate(ch_mult):
      self.down_block = []
      self.down_block.append(MultipleSequential(*[ResBlock(ch,self.emb_ch,ch,self.dropout)]*self.num_res))
      if height >= attention_at_height:
        self.down_block.append(AttentionBlock(ch,self.num_heads))
      self.down_block.append(Downsample(ch))
      self.down_block.append(ResBlock(ch,self.emb_ch,model_ch * mult))
      self.down_blocks.append(MultipleSequential(*self.down_block))
      ch = model_ch * mult

    self.bottle_neck = MultipleSequential(ResBlock(ch,self.emb_ch,ch,self.dropout),AttentionBlock(ch,self.num_heads),ResBlock(ch,self.emb_ch,ch,self.dropout))
    self.up_blocks = nn.ModuleList()
    for height, mult in enumerate((ch_mult[::-1]+(1,))[1:]):
      self.up_block = []
      self.up_block.append(MultipleSequential(*[ResBlock(ch,self.emb_ch,ch,self.dropout)]*self.num_res))
      if height <= len(ch_mult)-attention_at_height:
        self.up_block.append(AttentionBlock(ch,self.num_heads))
      self.up_block.append(Upsample(ch))
      self.up_block.append(ResBlock(ch,self.emb_ch,model_ch * mult))
      self.up_blocks.append(MultipleSequential(*self.up_block))
      ch = model_ch * mult

    self.out = nn.Sequential(nn.GroupNorm(num_groups=min(ch,32), num_channels=ch),
                             nn.SiLU(),
                             nn.Conv2d(ch,out_ch,kernel_size = 3, padding = 1))

  def forward(self,x,timesteps,rds,cars,xys):
    '''
    Unet to predict noise
    :param x: 4-D tensor. The shape should be (batch,in_ch,img_sz,img_sz)
    :param timesteps: 1-D tensor. The shape should be the same as batch size.
    :param rds: 2-D tensor. The shape should be (batch,1024)
    :param cars: 2-D tensor. The shape should be (batch,1024)
    :param xys: 2/3-D tensor. The shape should be (batch,32) or (batch,16,2)
    '''
    # if len(xys.shape)==3:
    #   xy_nums = xys.shape[-2]*xys.shape[-1]
    # elif len(xys.shape)==2:
    #   xy_nums = xys.shape[-1]
    # trcx_emb = T_R_C_X_Embedding(xy_nums=xy_nums,output_dim=self.model_ch)
    emb = self.emb_block(self.trcx_emb(timesteps,rds,cars,xys))

    x = self.inn(x)
    skip_connection = []
    for block in self.down_blocks:
      skip_connection.append(x)
      x = block(x,emb)

    x = self.bottle_neck(x,emb)

    for block in self.up_blocks:

      x = block(x,emb)
      x+= skip_connection.pop()

    x = self.out(x)

    return x
