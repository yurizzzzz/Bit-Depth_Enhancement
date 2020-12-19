from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


# 图像加载器
def image_loader(path):
    return Image.open(path).convert('RGB')


'''
自定义的数据加载类,根据Pytorch的数据导入规则,在自定义数据加载类的时候必须定义__getitem__和__len__方法
其中getitem实际上作为读入列表元素的作用并返回元素,而len的话是返回容器中元素的个数,要是没有在类中定义__len__
方法那么如果在后续创建好类后想要调用len()会发生报错,所以__len__的目的是为了让程序中的len执行成功
'''


class LoadData(Dataset):
    # args是在Train.py中要输入的各类参数, img输入实际上是一个包含数据文件名的列表
    # 注意：本代码中train和test数据集的命名都是1.jpg ·····等；label数据的命名是MITAdobe-1.jpg··等
    def __init__(self, args, img, loader=image_loader, mode='train'):
        self.img = img
        self.loader = loader
        self.mode = mode
        self.args = args

    # 遍历访问列表中文件名
    def __getitem__(self, item):
        input_img = self.img[item]

        # 创建torch的tensor实例使得普通的array转换成tensor
        t_list = [transforms.ToTensor()]
        composed_transform = transforms.Compose(t_list)

        if self.mode == 'train':
            image = self.loader(self.args.train_dir + '/' + input_img)
            label = self.loader(self.args.label_dir + '/' + 'MITAdobe-' + input_img)

            # 以图像中心为原点截取大小为256x256的图像作为训练输入数据
            center_crop_ = transforms.Compose([transforms.CenterCrop((256, 256))])
            image = center_crop_(image)
            label = center_crop_(label)

            image = composed_transform(image)
            label = composed_transform(label)

        elif self.mode == 'validate' or self.mode == 'test':
            # 测试验证数据集不作任何变换
            image = self.loader(self.args.test_dir + '/' + input_img)
            label = self.loader(self.args.label_dir + '/' + 'MITAdobe-' + input_img)

            image = composed_transform(image)
            label = composed_transform(label)

        else:
            print('ERROE')
            raise NotImplementedError

        # 返回4bit的输入图像和真实的8bit标签图像
        return image, label

    def __len__(self):
        return len(self.img)

    
