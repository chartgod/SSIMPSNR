#psnr.py

import numpy
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
import numpy
import math
import numpy as np
from PIL import Image
from psnr import psnr
from ssim import ssim, ssim_exact

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

##ssim.py
#2차원 이미지만 사용할 수 있도록 수정함. 다른차원도 가지고 있을 가능성이 있음. 그러니 틀린 내용임.

def block_view(A, block=(3, 3)):
    if A.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    shape = (A.shape[0] // block[0], A.shape[1] // block[1]) + block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    bimg1 = block_view(img1, (4,4))
    bimg2 = block_view(img2, (4,4))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)

# 원본 이미지와 변환된 이미지 불러오기
original_image = Image.open('C:/dcp-dehaze-master/image/output/make/hazy/dcp1.jpg')
converted_image = Image.open('C:/dcp-dehaze-master/image/output/make/dcp/dcp1.jpg')

# 이미지를 NumPy 배열로 변환
original_array = np.array(original_image)
converted_array = np.array(converted_image)

# 이미지를 NumPy 배열로 변환
original_array = np.array(original_image.convert('L'))
converted_array = np.array(converted_image.convert('L'))

# PSNR 계산
psnr_value = psnr(original_array, converted_array)
print(f"PSNR: {psnr_value:.2f}")

# SSIM 계산 (SSIM 또는 SSIM-Exact 중 선택)
ssim_value = ssim(original_array, converted_array)


print(f"SSIM: {ssim_value:.4f}")
