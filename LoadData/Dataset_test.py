from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
from time import *


def image_loader(path):
    return Image.open(path).convert('RGB')


class LoadData(Dataset):
    def __init__(self, img, loader=image_loader, mode='train'):
        self.img = img
        self.loader = loader
        self.mode = mode

    def __getitem__(self, item):
        input_img = self.img[item]

        t_list = [transforms.ToTensor()]
        composed_transform = transforms.Compose(t_list)

        if self.mode == 'train':
            image = self.loader('E:/train(512x512)' + '/' + input_img)
            label = self.loader('E:/label(512x512)' + '/' + 'MITAdobe-' + input_img)

            image = composed_transform(image)
            label = composed_transform(label)

        elif self.mode == 'validate' or self.mode == 'test':
            image = self.loader('E:/test(512x512)' + '/' + input_img)
            label = self.loader('E:/label(512x512)' + '/' + 'MITAdobe-' + input_img)

            image = composed_transform(image)
            label = composed_transform(label)

        else:
            print('ERROE')
            raise NotImplementedError

        return image, label

    def __len__(self):
        return len(self.img)


# 测试Dataset数据集加载类的可用性
if __name__ == '__main__':
    # 开始时间
    begin = time()
    # listdir中输入训练数据集的路径返回的data是一个包含各个图像文件名的列表
    data = [file for file in os.listdir('E:/train(512x512)') if file.endswith('.jpg')]
    # 调用数据加载类
    trainData = LoadData(data)
    # 调用torch官方的数据load函数并设置batch_size,shuffle是否打乱等参数
    trainImg = DataLoader(trainData, batch_size=10, shuffle=True)
    # 循环遍历训练数据和标签数据
    for step, (batch_x, batch_y) in enumerate(tqdm(trainImg)):
        print(batch_x)
