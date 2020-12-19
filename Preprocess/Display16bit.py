import numpy as np
import cv2
import imageio
import rawpy
import png

from PIL import Image
from osgeo import gdal
from skimage import io

# 本代码用于显示16位深度的png图像

raw = rawpy.imread('1.dng')
im = raw.postprocess(output_bps=16)
im = np.asarray(im, np.uint16)
b, g, r = cv2.split(im)
image = cv2.merge([r, g, b])
cv2.imshow('DST', image)
cv2.imwrite('1.png', image)
cv2.waitKey(0)


'''
img = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)

cv2.imshow('DST', img)
cv2.waitKey(0)
'''

"""
reader = png.Reader('test.png')
data = reader.asDirect()
pixels = data[2]
count = 0
image = []
for row in pixels:
    row = np.asarray(row)
    row = np.reshape(row, [-1, 3])
    image.append(row)

image = np.stack(image, 1)
print(image.dtype)
print(image.shape)
b, g, r = cv2.split(image)
image = cv2.merge([r, g, b])
"""




"""
lon_offset_px=0
lat_offset_px=0

gdo = gdal.Open('test.png')
band = gdo.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize
png_array = gdo.ReadAsArray(lon_offset_px, lat_offset_px, xsize, ysize)
png_array = np.array(png_array)
print(png_array.shape)
print(png_array.dtype)
print(png_array)
"""


