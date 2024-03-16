#dataset.py
import torch
from PIL import Image
from torchvision import datasets, transforms
import os
import csv
from torch.utils.data import DataLoader
class DiffusionDataset(torch.utils.data.Dataset):
  def __init__(self,root_dir,rd_transform=None,car_transform=None,max_car=16):
    self.root_dir = root_dir
    self.gt_dir_list = sorted([os.path.join(root_dir,"gt",i) for i in os.listdir(os.path.join(root_dir,"gt"))])
    self.emp_rd_csv_dir = os.path.join(root_dir,"emp_rd","emp_rd.csv")
    self.cars_csv_dir = os.path.join(root_dir,"cars","cars.csv")
    self.cars_dir_list = sorted([os.path.join(root_dir,"cars",i) for i in os.listdir(os.path.join(root_dir,"cars"))])
    self.max_car = max_car
    self.emp_rd_len = None
    self.cars_csv_len = None
    self.emp_rd_data_list = None
    self.cars_data_list = None


    if rd_transform is None:
      self.rd_transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    else:
      self.rd_transform = rd_transform

    if car_transform is None:
      self.car_transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    else:
      self.car_transform = car_transform

  def load_img(self,img_path):
    img = Image.open(img_path)
    return img

  def __len__(self):
    if self.emp_rd_len is None:
      with open(self.emp_rd_csv_dir,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        self.emp_rd_len = sum(1 for row in csv_reader)
    if self.cars_csv_len is None:
      with open(self.cars_csv_dir,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        self.cars_csv_len = sum(1 for row in csv_reader)
    gt_len = len(self.gt_dir_list)
    cars_len = len(self.cars_dir_list)-1
    assert gt_len == self.emp_rd_len == cars_len == self.cars_csv_len, "All sub-directory must contain the same number of files or directories while gt_len = {}, emp_rd_len={}, cars_csv_len={}, and cars_len={}".format(gt_len,self.emp_rd_len,self.cars_csv_len,cars_len)
    return gt_len

  def __getitem__(self,idx):

    gt_img_path = self.gt_dir_list[idx]
    gt_img = self.load_img(gt_img_path)
    gt_img = self.rd_transform(gt_img)
    if self.emp_rd_data_list is None:
      with open(self.emp_rd_csv_dir,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        self.emp_rd_data_list = [(lambda x:[float(i) for i in x])(i) for i in csv_reader]
    emp_rd_data=torch.tensor(self.emp_rd_data_list[idx])
    
    if self.cars_data_list is None:
      with open(self.cars_csv_dir,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        self.cars_data_list = [(lambda x:[float(i) for i in x])(i) for i in csv_reader]
    cars_data = torch.tensor(self.cars_data_list[idx])
    xys = torch.zeros((self.max_car,2),dtype=torch.int)
    xylens = torch.zeros((self.max_car,2),dtype=torch.int)
    # try:
    csvf = [file for file in os.listdir(self.cars_dir_list[idx]) if file[-4:]==".csv"][0]
    cars_csv_path = os.path.join(self.cars_dir_list[idx],csvf)
    # except:
    #   cars_csv_path = os.path.join(self.cars_dir_list[idx],"xy.csv")
    with open(cars_csv_path,'r') as csv_file:
      csv_reader = csv.reader(csv_file)
      for i,item in enumerate(csv_reader):
        car_img_path,x,y,x_length,y_length = item
        xy_tensor = torch.tensor((int(x)*256//640,int(y)*256//640),dtype=torch.int)
        xylen_tensor = torch.tensor((int(x_length)*256//640,int(y_length)*256//640),dtype=torch.int)
        car_img_path = os.path.join(self.cars_dir_list[idx],car_img_path)
        # car_img = self.load_img(car_img_path)
        # car_img = self.car_transform(car_img)

        xys[i] = xy_tensor
        xylens[i] = xylen_tensor
        # car_imgs[i] = car_img
      # print(gt_img.shape)
      # print(emp_rd_data.shape)
      # print(xys.shape)
      # print(cars_data.shape)
      return gt_img,emp_rd_data,xys,cars_data, xylens



if __name__ == "__main__":
  dataset = DiffusionDataset("/dataset")
  dataloader=DataLoader(dataset,batch_size=64,shuffle = True)
  for gt,emp,xy,car in dataloader:
    print(gt.shape,emp.shape,car.shape,xy.shape)
    print(car[1][0],xy[1][0])

