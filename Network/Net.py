import torch.nn as nn


# 网络中的图像数据的尺寸大小保持不变为了使更多的特征不被流失
#                        网络大体结构为
#                        conv1+Relu——conv2+Relu——conv3+Relu——conv4+Relu——conv5+Relu
#                                   |           |          |           |       |
#   deconv5+BN+Relu——deconv4+BN+Relu——deconv3+BN+Relu——deconv2+BN+Relu——deconv1+BN+Relu
class Model(nn.Module):
    def __init__(self):
        # 继承父类的一个方法
        super(Model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

    def forward(self, input_img):
        inputs = input_img
        # 设置连接层
        skip_connection = []

        img = self.conv1(inputs)
        skip_connection.append(img)

        img = self.conv2(img)
        skip_connection.append(img)

        img = self.conv3(img)
        skip_connection.append(img)

        img = self.conv4(img)
        skip_connection.append(img)

        img = self.conv5(img)

        img = self.deconv1(img)
        img = skip_connection.pop() + img

        img = self.deconv2(img)
        img = skip_connection.pop() + img

        img = self.deconv3(img)
        img = skip_connection.pop() + img

        img = self.deconv4(img)
        img = skip_connection.pop() + img

        img = self.deconv5(img)

        # 最后输出的img是网络真正输出的特征图像
        result = img + input_img

        return result

    
