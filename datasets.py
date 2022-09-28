from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torch.utils.data
import random
from os.path import join, splitext, basename
from glob import glob
import math
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop

def rand_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and random crop to target size

    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = random.randint(0, width - target_width)
        starting_y = random.randint(0, height - target_height)
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = random.randint(0, new_width - target_width)
        starting_y = random.randint(0, new_height - target_height)
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

def center_crop(img, target_height, target_width):
    # reshape image to an appropriate size, and center crop to target size
    
    width = img.size[0]
    height = img.size[1]

    width_scale = target_width / width
    height_scale = target_height / height
    
    if height_scale >= 0.5:
        starting_x = (width - target_width) / 2
        starting_y = (height - target_height) / 2
    else:
        down_sample_ratio = height_scale / 0.5
        if round(down_sample_ratio*width) < target_width:
            down_sample_ratio = width_scale
        new_width = round(down_sample_ratio * width)
        new_height = round(down_sample_ratio * height)
        img = img.resize((new_width, new_height)) 
        starting_x = (new_width - target_width) / 2
        starting_y = (new_height - target_height) / 2
        
    img = img.crop((starting_x, starting_y, starting_x+target_width, starting_y+target_height))
    
    return img

class dataset_norm(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        #self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.inputsize = inputsize

        self.img_list = sorted(glob(join(root, '*.jpg')))

        # for name in file_list:
        #     img = Image.open(name)
        #     if (img.size[0] >= (self.imgSize)) and (img.size[1] >= self.imgSize):
        #         self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        img = Image.open(self.img_list[index])
        i = (self.imgSize - self.inputsize)//2

        img = self.transforms(img)

        iner_img = img[:, i:i + self.inputsize, :]
        iner_img = iner_img[:, :, i:i+self.inputsize]
        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,i:i + self.inputsize,i:i + self.inputsize] = iner_img

        return img, mask_img

    def __len__(self):
        return self.size

class dataset_test4(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize
        self.inputsize2 = inputsize + 64 * (pred_step - 1)

        self.img_list = sorted(glob(join(root, '*.jpg')))

        # for name in file_list:
        #     img = Image.open(name)
        #     #if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
        #     if (img.size[0] >= 0) and (img.size[1] >= 0):
        #         self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2
        #j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        iner_img = img[:, i:i + self.inputsize, :]
        iner_img = iner_img[:, :, i:i+self.inputsize]
        #mask_img = np.zeros((3, self.imgSize, self.imgSize))
        #mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        #mask_img[:, i:i + self.inputsize, i:i+self.inputsize] = iner_img
        mask_img = np.ones((3, self.preSize, self.preSize))
        if self.pred_step > 1:
        #mask_img[:,i:i + self.inputsize + 64*(self.pred_step-1),i:i + self.inputsize + 32*self.pred_step]=img
            mask_img[:, i:i + self.inputsize2, i:i+self.inputsize2] = img
        else:
            mask_img[:, i:i + self.inputsize2, i:i + self.inputsize2] = iner_img

        return img, iner_img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_test3(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, crop='center', rand_pair=False, imgSize=192, inputsize=128):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.transforms = transforms
        self.imgSize = imgSize
        self.rand_pair = rand_pair
        self.inputsize = inputsize
        if crop == 'rand':
            self.cropFunc = rand_crop
        elif crop == 'center':
            self.cropFunc = center_crop
        else:
            self.cropFunc = None

        file_list = sorted(glob(join(root, '*.jpg')))

        for name in file_list:
            img = Image.open(name)
            if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
                self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2

        if self.cropFunc is not None:
            img = self.cropFunc(img, self.imgSize, self.imgSize)

        if self.transforms is not None:
            img = self.transforms(img)

        iner_img = img[:, i:i + self.inputsize, :]
        iner_img = iner_img[:, :, i:i+self.inputsize]
        mask_img = np.zeros((3, self.imgSize, self.imgSize))
        mask_img[:, i:i + self.inputsize, i:i + self.inputsize] = iner_img
        #mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        #mask_img = img * mask
        #mask_img = np.ones((3, self.imgSize, self.imgSize))
        #mask_img[:, i:i + self.inputsize, i:i + self.inputsize] = iner_img

        return img, iner_img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):
        # --PARAMS--
        # root: the path of the data
        # crop: 'rand' or 'center' or 'none', which way to crop the image into target size
        # imgSize: the size of the returned image if crop is not 'none'

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize

        file_list = sorted(glob(join(root, '*.png')))

        for name in file_list:
            img = Image.open(name)
            # if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
            if (img.size[0] >= 0) and (img.size[1] >= 0):
                self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2
        j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        #iner_img = img[:, i:i + self.inputsize, :]
        #iner_img = iner_img[:, :, i:i + self.inputsize]
        # mask_img = np.zeros((3, self.imgSize, self.imgSize))
        # mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        # mask_img[:, i:i + self.inputsize, i:i+self.inputsize] = iner_img
        mask_img = np.ones((3, self.preSize, self.preSize))
        if self.pred_step > 1:
            # mask_img[:,i:i + self.inputsize + 64*(self.pred_step-1),i:i + self.inputsize + 32*self.pred_step]=img
            mask_img[:, i:i + self.imgSize, i:i + self.imgSize] = img
        else:
            mask_img[:, i:i + self.inputsize2, i:i + self.inputsize2] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi2(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        #self.preSize = imgSize + 64 * (pred_step - 1)
        self.inputsize = inputsize
        #self.cropsize = int(imgSize//(1.5**pred_step))

        self.img_list = sorted(glob(join(root, '*.jpg')))

        # for name in file_list:
        #     img = Image.open(name)
        #     # if (img.size[0] >= self.imgSize) and (img.size[1] >= self.imgSize):
        #     if (img.size[0] >= 0) and (img.size[1] >= 0):
        #         self.img_list += [name]

        self.size = len(self.img_list)

    def __getitem__(self, index):
        # --RETURN--
        # input1(left), input2(right), groundtruth of the intermediate region

        index = index % self.size
        name = self.img_list[index]
        img = Image.open(name).convert('RGB')
        i = (self.imgSize - self.inputsize) // 2
        #j = (self.preSize - self.inputsize) // 2

        if self.transforms is not None:
            img = self.transforms(img)

        #crop = img[:,]

        iner_img = img[:, i:i + self.inputsize, :]
        iner_img = iner_img[:, :, i:i + self.inputsize]
        crop_img = Resize(128)(iner_img)
        # mask_img = np.zeros((3, self.imgSize, self.imgSize))
        # mask[:, i:i + self.inputsize, i:i+self.inputsize] = 1
        # mask_img[:, i:i + self.inputsize, i:i+self.inputsize] = iner_img
        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = crop_img

        return img, crop_img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi3(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.file_list = sorted(glob(join(root, '*.png')))

        self.size = len(self.file_list)

    def __getitem__(self, index):

        index = index % self.size
        name = self.file_list[index]
        img = Image.open(name).convert('RGB')


        if self.transforms is not None:
            img = self.transforms(img)

        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size

class dataset_arbi4(Dataset):
    # prepare data for self-reconstruction,
    # where the two input photos and the intermediate region are obtained from the same image

    def __init__(self, root='', transforms=None, imgSize=192, inputsize=128, pred_step=1):

        self.img_list = []
        self.pred_step = pred_step
        self.transforms = transforms
        self.imgSize = imgSize
        self.file_list = sorted(glob(join(root, '*.jpg')))

        self.size = len(self.file_list)

    def __getitem__(self, index):

        index = index % self.size
        name = self.file_list[index]
        img = Image.open(name)


        if self.transforms is not None:
            img = self.transforms(img)

        mask_img = np.ones((3, self.imgSize, self.imgSize))
        mask_img[:,32:160,32:160] = img

        return img, img, mask_img, splitext(basename(name))[0]

    def __len__(self):
        return self.size
