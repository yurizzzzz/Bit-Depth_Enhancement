import sys
import os

# 在终端直接python train.py的时候需要加上下面这段代码使得其他包能够加入到sys.path中
if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))
import torch
import argparse
import torch.utils.data as Data
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision import utils as vutils
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from Network import Net
from LoadData import Dataset
from tqdm import tqdm
import cv2
from skimage.measure import compare_psnr, compare_ssim
import numpy as np


# 计算PSNR的值
def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)


# 计算SSIM的值
def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


# 输入参数
def input_args():
    parser = argparse.ArgumentParser()

    # 模型存放位置
    parser.add_argument("--model_dir", type=str, default='/home/model',
                        help="Location at which to save the model, logs and checkpoints.")
    # 训练数据目录
    parser.add_argument("--train_dir", type=str, default='/home/train512',
                        help="Directory containing source JPG images for training.")
    # 测试数据目录
    parser.add_argument("--test_dir", type=str, default='/home/test512',
                        help="Directory containing source JPG images for testing/validating.")
    # 标签数据目录
    parser.add_argument("--label_dir", type=str, default='/home/label512',
                        help="Directory containing source JPG images for labeling.")
    # 设置batch_size
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size to train the model.")
    # 设置周期数
    parser.add_argument("--epochs", type=int, default=200,
                        help="No of epochs to train and validate the model.")
    parser.add_argument("--epoch_start", type=int, default=0,
                        help="Epoch to start training the model from.")
    # 设置Adam的学习率
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for the model.")
    # 设置GPU_ID首选
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='Select the gpu_id to run the code on')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Whether to skip validation in the training process?')

    return parser.parse_args()


# 训练数据函数
def train_data(model, optimizer, inputs, labels):
    # 启动模型训练
    model.train()

    # 将输入的tensor数据转换到GPU的CUDA上
    sRGB_input = Variable(torch.FloatTensor(inputs)).cuda()
    sRGB_label = Variable(torch.FloatTensor(labels)).cuda()

    # 将一轮batch的梯度置零，如果不置零那么梯度值将会进入到下一个batch相当于提高了batch_size使得内存增加
    optimizer.zero_grad()
    sRGB_output = model(sRGB_input)

    # 反向传播
    loss = F.l1_loss(sRGB_output, sRGB_label.detach())
    loss.backward()
    optimizer.step()

    return loss, sRGB_input, sRGB_output, sRGB_label


# 测试数据
def validate_data(model, inputs, labels):
    # 停止模型的梯度计算，反向传播等
    model.eval()

    # 将输入数据转移到CUDA上计算
    sRGB_input = Variable(torch.FloatTensor(inputs)).cuda()
    sRGB_label = Variable(torch.FloatTensor(labels)).cuda()

    # 停止计算梯度，测试数据不需要计算梯度同时也是为了节省内存空间
    with torch.no_grad():
        # 得出模型输出结果
        sRGB_output = model(sRGB_input)

    # 将输出的tensor转换成array矩阵形式为了下面PSNR和SSIM的计算
    sRGB_output = sRGB_output[0, :, :, :].cpu().data.numpy().transpose((1, 2, 0))
    sRGB_output = np.array(sRGB_output * 255.0, dtype='uint8')
    sRGB_label = sRGB_label[0, :, :, :].cpu().data.numpy().transpose((1, 2, 0))
    sRGB_label = np.array(sRGB_label * 255.0, dtype='uint8')

    # 计算PSNR和SSIM
    cur_psnr = calc_psnr(sRGB_output, sRGB_label)
    cur_ssim = calc_ssim(sRGB_output, sRGB_label)

    return cur_psnr, cur_ssim


if __name__ == '__main__':
    args = input_args()
    args.gpu_id = 0
    args.logs_dir = args.model_dir + '/logs/'
    args.visuals_dir = args.model_dir + '/visuals/'
    args.nets_dir = args.model_dir + '/nets/'
    # 设置G要选择的GPU device
    torch.cuda.set_device(args.gpu_id)

    # 记录参数到tensorboardX中
    logger = SummaryWriter(args.logs_dir)

    # 创建网络模型
    model = Net.Model().cuda()
    model = nn.DataParallel(model, device_ids=[args.gpu_id]).cuda()

    # 创建优化器，采用Adam优化方法
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    # 读取训练数据和测试数据
    train_list = [file for file in os.listdir(args.train_dir) if file.endswith('.jpg')]
    validate_list = [file for file in os.listdir(args.test_dir) if file.endswith('.jpg')]

    # 调用torch载入数据集
    TrainData = torch.utils.data.DataLoader(Dataset.LoadData(args, train_list, mode='train'),
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=8,
                                            drop_last=False)
    ValidateData = torch.utils.data.DataLoader(Dataset.LoadData(args, validate_list, mode='validate'),
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=8,
                                               drop_last=False)

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_tr_loss = 0.0

    # 在训练数据前开始validate测试初始状态下的PSNR和SSIM指标
    if not args.skip_validation:
        epoch = args.epoch_start
        cumulative_psnr = 0
        cumulative_ssim = 0
        count_idx = 0

        tbar = tqdm(ValidateData)

        for batch_idx, (inputs, labels) in enumerate(tbar):
            count_idx = count_idx + 1

            cur_psnr, cur_ssim = validate_data(model, inputs, labels)
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
            avg_psnr = cumulative_psnr / count_idx
            avg_ssim = cumulative_ssim / count_idx
            desc = 'Validate: Epoch %d, PSNR = %.4f and SSIM = %.4f' % (epoch, avg_psnr, avg_ssim)
            tbar.set_description(desc)
            tbar.update()

        logger.add_scalar('Validation/avg_psnr', avg_psnr, epoch)
        logger.add_scalar('Validation/avg_ssim', avg_ssim, epoch)

    # 训练数据开始
    giters = 0
    # 循环设定的周期数
    for epoch in range(args.epoch_start + 1, args.epoch_start + args.epochs + 1):
        # 初始化要记录的参数值
        tr_loss = 0
        cumulative_psnr = 0
        cumulative_ssim = 0
        count_idx = 0
        # tqdm是进度条包，使得在代码运行的时候更能直观的看到运行情况
        tbar = tqdm(TrainData)
        # 遍历训练数据和标签数据
        for batch_idx, (inputs, labels) in enumerate(tbar):
            count_idx = count_idx + 1

            # 返回损失值，输入数据，输出数据，标签数据
            loss, rgb_input, rgb_output, rgb_label = train_data(model, optimizer, inputs, labels)
            tr_loss = tr_loss + loss
            logger.add_scalar('Train/loss', loss, giters)
            # 每100次保存一次图像数据
            if giters % 100 == 0:
                input_save = rgb_input.detach().cpu()
                output_save = rgb_output.detach().cpu()
                label_save = rgb_label.detach().cpu()
                result_save = torch.cat((input_save, output_save, label_save), 2)
                vutils.save_image(result_save, args.visuals_dir + '/visual' + str(epoch) + '_' + str(giters) + '.jpg')
            giters = giters + 1
            avg_tr_loss = tr_loss / count_idx
            desc = 'Training  : Epoch %d, Avg. Loss = %.5f' % (epoch, avg_tr_loss)
            tbar.set_description(desc)
            tbar.update()

        logger.add_scalar('Train/avg_loss', avg_tr_loss, epoch)

        # 测试数据计算指标
        if not args.skip_validation:
            count_idx = 0
            tbar = tqdm(ValidateData)
            for batch_idx, (inputs, labels) in enumerate(tbar):
                count_idx = count_idx + 1

                cur_psnr, cur_ssim = validate_data(model, inputs, labels)
                cumulative_psnr += cur_psnr
                cumulative_ssim += cur_ssim
                avg_psnr = cumulative_psnr / count_idx
                avg_ssim = cumulative_ssim / count_idx
                desc = 'Validation: Epoch %d, Avg. PSNR = %.4f and SSIM = %.4f' % (epoch, avg_psnr, avg_ssim)
                tbar.set_description(desc)
                tbar.update()

            logger.add_scalar('Validation/avg_psnr', avg_psnr, epoch)
            logger.add_scalar('Validation/avg_ssim', avg_ssim, epoch)

            # 保存每一个epoch 的模型参数数据和指标
            savefilename = args.nets_dir + '/checkpoint' + '_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'state_dict': model.state_dict(),
                'avg_tr_loss': avg_tr_loss}, savefilename)

            
