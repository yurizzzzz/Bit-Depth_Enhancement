from time import time
from tqdm import tqdm
import cv2

# 高低位深度，本代码中是将图像从8bit转换成4bit，量化使用均匀量化
high_bit = 8
low_bit = 4
cal = 2 ** (high_bit - low_bit)


begin = time()

bar = [x for x in range(0, 1)]
bar = tqdm(bar)

for index in bar:
    img = cv2.imread('C://Users//dell//Desktop//' + str(index + 1) + '.jpg')
    img = img // cal * cal

    cv2.imwrite('C://Users//dell//Desktop//data//' + str(index + 1) + '.jpg', img)


end = time()
cost = end - begin
print('Run time: ', cost)

