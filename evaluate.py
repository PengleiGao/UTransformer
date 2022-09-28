import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop
import os
from os.path import join
from models.build4 import build_model
#from models.swin_transformer4 import build_model
from dataset import dataset_test4, dataset_arbi2, dataset_arbi3, dataset_arbi
#from dataset import dataset_test3
#from dataset import dataset_arbi
import argparse
import skimage
from skimage import io
from scipy.ndimage.morphology import distance_transform_edt
import skimage.transform
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Evaluate
def evaluate(gen, eval_loader, rand_pair, save_dir):
    gen.eval()

    os.makedirs(join(save_dir, 'result'), exist_ok=True)
    os.makedirs(join(save_dir, 'blend'), exist_ok=True)

    if rand_pair:
        os.makedirs(join(save_dir, 'input'), exist_ok=True)
        # os.makedirs(join(save_dir, 'input2'), exist_ok=True)

    com_total = 0
    for batch_idx, (gt, iner_img, mask_img, name) in enumerate(eval_loader):

        imgSize = gt.shape[2]

        gt, iner_img, mask_img = Variable(gt).cuda(), Variable(iner_img).cuda(), Variable(mask_img.type(torch.FloatTensor)).cuda()
        with torch.no_grad():
            t_start = time()
            #I_pred, _ = gen(gt)
            I_pred, _ = gen(mask_img)
            t_end = time()
            comsum = t_end - t_start
            com_total += comsum
            #I_pred[:,:,32:128+32,32:128+32] = iner_img

        for i in range(gt.size(0)):

            blended_img = np.transpose(I_pred[i].data.cpu().numpy(), (1, 2, 0))
            pre_img = np.transpose(I_pred[i].data.cpu().numpy(), (1, 2, 0))
            std_ = np.expand_dims(np.expand_dims(np.array(std), 0), 0)
            mean_ = np.expand_dims(np.expand_dims(np.array(mean), 0), 0)
            real = np.transpose(gt[i].data.cpu().numpy(), (1, 2, 0))
            real = real * std_ + mean_
            real = np.clip(real, 0, 1)

            iner = np.transpose(iner_img[i].data.cpu().numpy(), (1, 2, 0))

            iner = iner * std_ + mean_
            iner = np.clip(iner, 0, 1)
            io.imsave(join(save_dir, 'input', '%s.png' % (name[i])), skimage.img_as_ubyte(real))

            #print(mean_.shape())
            pre_img = pre_img * std_ + mean_
            pre_img = np.clip(pre_img, 0, 1)

            #blended_img, src_mask = blend_result(pre_img, iner)
            #blended_img = pre_img
            #blended_img[32:160, 32:160, :] = iner

            # blended_img = blended_img * std_ + mean_
            # blended_img = np.clip(blended_img, 0, 1)

            io.imsave(join(save_dir, 'result', '%s.png' % (name[i])), skimage.img_as_ubyte(pre_img))
            #io.imsave(join(save_dir, 'blend', '%s.png' % (name[i])), skimage.img_as_ubyte(blended_img))

        #break
    avg = com_total / len(eval_loader)
    print(avg)

if __name__ == '__main__':

    LOAD_WEIGHT_DIR = 'path of your checkpoint'
    TEST_DATA_DIR = 'path to test data'
    SAVE_DIR = 'path to save result'




    def get_args():

        parser = argparse.ArgumentParser()

        parser.add_argument('--rand_pair', type=bool, help='pair testing data randomly', default=True)

        # parser.add_argument('--skip_connection', type=int,help='layers with skip connection', nargs='+', default=[0,1,2,3,4])
        # parser.add_argument('--attention', type=int,help='layers with attention mechanism applied on skip connection', nargs='+', default=[1])

        parser.add_argument('--load_weight_dir', type=str, help='directory of pretrain model weights',
                            default=LOAD_WEIGHT_DIR)
        parser.add_argument('--save_dir', type=str, help='directory of saving results', default=SAVE_DIR)
        parser.add_argument('--test_data_dir', type=str, help='directory of testing data 1', default=TEST_DATA_DIR)
        # parser.add_argument('--test_data_dir_2',type=str,help='directory of testing data 2',default=TEST_DATA_DIR_2)
        parser.add_argument('--adjoint', type=bool, help='if use adjoint in odenet', default=True)

        opts = parser.parse_args()
        return opts


    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    pred_step = 1
    times = 1
    input_size = [128,86,56,38]
    #mean = [0.504, 0.513, 0.521]
    #std = [0.241, 0.222, 0.259]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    config = {}
    config['pre_step'] = pred_step
    config['TYPE'] = 'swin'
    config['IMG_SIZE'] = 224
    config['SWIN.PATCH_SIZE'] = 4
    config['SWIN.IN_CHANS'] = 3
    config['SWIN.EMBED_DIM'] = 96
    config['SWIN.DEPTHS'] = [2, 2, 6, 2]
    config['SWIN.NUM_HEADS'] = [3, 6, 12, 24]
    config['SWIN.WINDOW_SIZE'] = 7
    config['SWIN.MLP_RATIO'] = 4.
    config['SWIN.QKV_BIAS'] = True
    config['SWIN.QK_SCALE'] = None
    config['DROP_RATE'] = 0.0
    config['DROP_PATH_RATE'] = 0.2
    config['SWIN.PATCH_NORM'] = True
    config['TRAIN.USE_CHECKPOINT'] = False

    # Initialize the model
    print('Initializing model...')
    gen = build_model(config).cuda(0)

    # Load pre-trained weight
    print('Loading model weight...')
    gen.load_state_dict(torch.load(join(args.load_weight_dir, 'Gen_former_200')))

    # Load data
    print('Loading data...')
    if args.rand_pair:
        transformations = transforms.Compose([Resize(192),CenterCrop(192),ToTensor(), Normalize(mean, std)])
        #transformations = transforms.Compose([Resize(128), CenterCrop(128),ToTensor(), Normalize(mean, std)])
        eval_data = dataset_arbi(root=args.test_data_dir, transforms=transformations,
                                  imgSize=192, inputsize=input_size[times-1], pred_step=pred_step)
        # transformations = transforms.Compose([Resize((256,256)), ToTensor()])
        # eval_data = dataset_test(root1=args.test_data_dir_1, root2=args.test_data_dir_2, transforms=transformations, crop='none', rand_pair=False)
    eval_loader = DataLoader(eval_data, batch_size=100, shuffle=False)
    print('test data: %d image pairs' % (len(eval_loader.dataset)))

    # Evaluate
    evaluate(gen, eval_loader, args.rand_pair, args.save_dir)
