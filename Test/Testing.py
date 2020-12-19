import sys
import os

# 在终端直接python train.py的时候需要加上下面这段代码使得其他包能够加入到sys.path中
if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))

import torch
import argparse
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision import utils as vutils
import torch.nn as nn
from Network import Net
import PIL.Image as Image


def get_arguments():
    parser = argparse.ArgumentParser()

    # 训练好的模型存放目录
    parser.add_argument("--load_model", type=str, required=False,
                        default='C://Users//dell//Desktop/checkpoint_200.tar',
                        help="Location from which any pre-trained model needs to be loaded.")
    # 测试数据的目录
    parser.add_argument("--data_dir", type=str, required=False, default='C://Users//dell//Desktop/data',
                        help="Directory containing the Darmstadt RAW images.")
    # 结果数据存放目录
    parser.add_argument("--results_dir", type=str, required=False, default='C://Users//dell//Desktop/result',
                        help="Directory to store the results in.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Select the args.gpu_id to run the code on')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_arguments()

    args.gpu_id = 0

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # 创建神经网络model模型
    torch.cuda.set_device(args.gpu_id)
    model = Net.Model().cuda()
    model = nn.DataParallel(model, device_ids=[args.gpu_id]).cuda()

    if args.load_model is not None:
        # 将先前训练好的模型参数导入刚刚创建好的模型中去
        model_psnr = torch.load(args.load_model, map_location='cuda:0')['avg_psnr']
        model_ssim = torch.load(args.load_model, map_location='cuda:0')['avg_ssim']
        model_state_dict = torch.load(args.load_model, map_location='cuda:0')['state_dict']
        model.load_state_dict(model_state_dict)

    # 读取测试数据并转换成tensor类型
    data_list = [file for file in os.listdir(args.data_dir) if file.endswith('.jpg')]
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)

    # 遍历数据
    for img in data_list:
        input_data = Image.open(args.data_dir + '/' + img).convert('RGB')
        input_data = composed_transform(input_data)

        # 由于每次只输入一张图像所以batch_size是可以设置成1但是输入的数据是3维即[channel, C, H]
        # 因此需要在输入数据的增添第一维使之成为四维数据
        input_data = input_data.unsqueeze(0)

        # 数据转到CUDA上进行计算
        input_data = Variable(torch.FloatTensor(input_data)).cuda()
        with torch.no_grad():
            output_data = model(input_data)

        # 输出tensor数据先转到CPU上在进行保存，为了增加对比性，将输入数据和输出数据在行上合并
        output_data = output_data.detach().cpu()
        input_data = input_data.detach().cpu()
        result = torch.cat((input_data, output_data), 2)
        vutils.save_image(result, args.results_dir + '/' + img)

        
