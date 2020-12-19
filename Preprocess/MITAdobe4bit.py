import cv2
import numpy as np
from time import *
from tqdm import tqdm

# 本代码中量化使用的是Kmeans非均匀量化，设置要转换的位数然后设置聚类簇数
begin = time()

bar = [x for x in range(3800, 4000)]
bar = tqdm(bar)

for index in bar:
    IMG = cv2.imread('E://MITdataset//MITAdobe-' + str(index + 1) + '.jpg')

    data = IMG.reshape((-1, 3))
    data = np.float32(data)

    # 设置K-means的阈值
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置随机初始化质心
    flags = cv2.KMEANS_RANDOM_CENTERS
    # K-means分类成16个簇
    compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)

    centers16 = np.uint8(centers16)
    res = centers16[labels16.flatten()]
    dst16 = res.reshape(IMG.shape)

    cv2.imwrite('E://MITAdobe4bit//' + str(index + 1) + '.jpg', dst16)

end = time()
cost = end - begin
print('Run time: ', cost)
