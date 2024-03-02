import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.system('python -m pytorch_fid /predict /input')
from cal_IS import inception_score
import numpy as np
from PIL import Image
import torch

# imgs=[]
# gt = '/input' #path of the input images
# fake = '/predict' #path of the generated images
# for f in os.listdir(fake):
#     im= np.array(Image.open(os.path.join(fake, f))).transpose(2, 0, 1).astype(np.float32)[:3]
#     im/=255
#     im=im*2-1
#     imgs.append(im)
# imgs=np.stack(imgs,0)
# imgs=torch.from_numpy(imgs).cuda()
# iscore=inception_score(imgs,cuda=True,batch_size=32,resize=True,splits=1)
# print('IS', iscore)
#
#
# from skimage.metrics import peak_signal_noise_ratio,structural_similarity
# from skimage.color import rgb2ycbcr
# import skimage.io as io
#
# ssim=[]
# psnr=[]
# for f1,f2 in zip(os.listdir(gt),os.listdir(fake)):
#     im_gt = io.imread(os.path.join(gt, f1))[:,:,0:3]
#     im_pred = io.imread(os.path.join(fake, f1.replace(gt,fake)))[:,:,0:3]
#     im_gt = im_gt / 255.0
#     im_pred = im_pred / 255.0
#     im_gt = rgb2ycbcr(im_gt)[:, :, 0:1]
#     im_pred = rgb2ycbcr(im_pred)[:, :, 0:1]
#     im_gt = im_gt / 255.0
#     im_pred = im_pred / 255.0
#     psnr.append(peak_signal_noise_ratio(im_gt,im_pred))
#     ssim.append(structural_similarity(im_gt,im_pred, win_size=11,gaussian_weights=True,multichannel=True,data_range=1.0,K1=0.01,K2=0.03,sigma=1.5))
#
# print('psnr:',np.mean(psnr))
# print('ssim:',np.mean(ssim))
