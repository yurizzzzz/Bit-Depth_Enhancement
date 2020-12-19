import torchvision.transforms as transforms
import PIL.Image as Image
import os
from tqdm import tqdm

# 本代码用于裁剪数据集图像
data = [file for file in os.listdir('E:/MITdataset') if file.endswith('.jpg')]
center_crop_ = transforms.Compose([transforms.CenterCrop((1024, 1024))])
data = tqdm(data)

for i in data:
    img = Image.open('E:/MITdataset/'+i)
    img = center_crop_(img)
    img.save('E:/label(1024x1024)/'+i)
