# Bit-Depth Enhancement(Pytorch Code)
Restore the low bit-depth images back to the high bit-depth images

The neural network is based on the UNet network

## Introduction
The task of bit depth enhancement is to recover the significant bits lost by quantization. It has important applications in high bit depth display and photo editing. Although most displays are 8-bit, many TVs and smartphones (such as Samsung Galaxy S10 and iPhone x) already support 10 bit displays to meet high dynamic range standards due to growing consumer demand for finer hue values. However, these displays are not fully utilized because most of the available image and video content is still 8-bit. If the 8-bit data is directly stretched on the 10 bit display, there will be obvious contour artifacts, color distortion and detail loss. Therefore, it is of great significance to study bit depth enhancement.

## Requirement
- NVIDIA GPU
