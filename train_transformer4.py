import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop
import os
from os.path import join
from models.build4 import build_model, ImagePool
#from models.Generator_former import Generator_former
from utils.loss import IDMRFLoss
from models.Discriminator_ml import MsImageDis
#from utils.utils import gaussian_weight
from tensorboardX import SummaryWriter
from dataset import dataset_norm
import argparse
from datetime import datetime
from torch.utils.data import Dataset,DataLoader,TensorDataset
# this version is with normlized input with mean and std, all layers are normalized,
# change the order of the 'make_layer' with norm-activate-conv,and use the multi-scal D
# use two kind feature, horizon and vertical

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Training
def train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer):
    gen.train()
    dis.train()

    #mse = nn.MSELoss().cuda(0)
    mae = nn.L1Loss().cuda(0)
    mrf = IDMRFLoss(device=0)

    acc_pixel_rec_loss = 0
    acc_feat_rec_loss = 0
    acc_mrf_loss = 0
    acc_feat_cons_loss = 0
    acc_gen_adv_loss = 0
    acc_dis_adv_loss = 0

    for batch_idx, (gt, mask_img) in enumerate(train_loader):

        batchSize = mask_img.shape[0]
        imgSize = mask_img.shape[2]

        #gt, mask_img, iner_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0), Variable(iner_img).cuda(0)
        gt, mask_img = Variable(gt).cuda(0), Variable(mask_img.type(torch.FloatTensor)).cuda(0)
        iner_img = gt[:, :, 32:32 + 128, 32:32 + 128]
        #I_groundtruth = torch.cat((I_l, I_r), 3)  # shape: B,C,H,W

        ## Generate Image
        I_pred, f_de = gen(mask_img)
        #I_pred = gen(mask_img)
        f_en = gen(iner_img, only_encode=True)

        # i_mask = torch.ones_like(gt)
        # i_mask[:, :, 32:32 + 128, 32:32 + 128] = 0
        # mask_pred = I_pred * i_mask
        mask_pred = I_pred[:, :, 32:32 + 128, 32:32 + 128]


        ## Compute losses
        ## Update Discriminator
        opt_dis.zero_grad()
        dis_adv_loss = dis.calc_dis_loss(I_pred.detach(), gt)
        dis_loss = dis_adv_loss
        dis_loss.backward()
        opt_dis.step()

        # Pixel Reconstruction Loss
        #pixel_rec_loss = mse(I_pred, gt) * 10
        #pixel_rec_loss = mse(mask_pred, iner_img) * 10
        pixel_rec_loss = mae(I_pred, gt) * 20

        # Texture Consistency Loss (IDMRF Loss)
        #mrf_loss = mrf(((I_pred * img_mask).cuda(0) + 1) / 2.0, ((gt * img_mask).cuda(0) + 1) / 2.0) * 0.01 / batchSize
        mrf_loss = mrf((mask_pred.cuda(0) + 1) / 2.0, (iner_img.cuda(0) + 1) / 2.0) * 0.5 / batchSize
        # mrf_loss = mrf((I_pred_split[1].cuda(1)+1)/2.0, (I_m.cuda(1)+1)/2.0) * 0.01

        # Feature Reconstruction Loss
        #feat_rec_loss = mse(f_all, f_all_gt.detach()).mean() * batchSize
        #feat_rec_loss = mse(f_all, f_all_gt.detach()) * 5
        feat_rec_loss = mae(f_de, f_en.detach())

        # ## Update Generator
        gen_adv_loss = dis.calc_gen_loss(I_pred, gt)
        #gen_loss = pixel_rec_loss + gen_adv_loss + mrf_loss.cuda(0) + feat_rec_loss
        gen_loss = pixel_rec_loss + gen_adv_loss + feat_rec_loss + mrf_loss.cuda(0)
        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

        acc_pixel_rec_loss += pixel_rec_loss.data
        acc_gen_adv_loss += gen_adv_loss.data
        acc_mrf_loss += mrf_loss.data
        acc_feat_rec_loss += feat_rec_loss.data
        #acc_feat_cons_loss += feat_cons_loss.data
        acc_dis_adv_loss += dis_adv_loss.data

        if batch_idx % 10 == 0:
            print("train iter %d" % batch_idx)
            print('generate_loss:', gen_loss.item())
            print('dis_loss:', dis_loss.item(

            ))

    ## Tensor board
    writer.add_scalars('train/generator_loss',
                       {'Pixel Reconstruction Loss': acc_pixel_rec_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Texture Consistency Loss': acc_mrf_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/generator_loss',
                       {'Feature Reconstruction Loss': acc_feat_rec_loss / len(train_loader.dataset)}, epoch)
    #writer.add_scalars('train/generator_loss', {'Feature Consistency Loss': acc_feat_cons_loss / len(train_loader.dataset)}, epoch)
    writer.add_scalars('train/generator_loss', {'Adversarial Loss': acc_gen_adv_loss / len(train_loader.dataset)},
                       epoch)
    writer.add_scalars('train/discriminator_loss', {'Adversarial Loss': acc_dis_adv_loss / len(train_loader.dataset)},
                       epoch)



if __name__ == '__main__':

    SAVE_WEIGHT_DIR = '/outpainting/checkpoints/former_resize_4-3/'
    SAVE_LOG_DIR = '/outpainting/logs_all/logs_former_resize_4-3/'
    TRAIN_DATA_DIR = '/outpainting/data3/train_all.asd'





    def get_args():

        parser = argparse.ArgumentParser()

        parser.add_argument('--train_batch_size', type=int, help='batch size of training data', default=40)
        parser.add_argument('--test_batch_size', type=int, help='batch size of testing data', default=16)
        parser.add_argument('--epochs', type=int, help='number of epoches', default=300)
        parser.add_argument('--lr', type=float, help='learning rate', default=2e-4)
        parser.add_argument('--alpha', type=float, help='learning rate decay for discriminator', default=0.1)
        parser.add_argument('--load_pretrain', type=bool, help='load pretrain weight', default=True)
        parser.add_argument('--test_flag', type=bool, help='testing while training', default=False)
        parser.add_argument('--adjoint', type=bool, help='if use adjoint in odenet', default=True)

        # parser.add_argument('--skip_connection', type=int,help='layers with skip connection', nargs='+', default=[0,1,2,3,4])
        # parser.add_argument('--attention', type=int,help='layers with attention mechanism applied on skip connection', nargs='+', default=[1])

        parser.add_argument('--load_weight_dir', type=str, help='directory of pretrain model weights',
                            default=LOAD_WEIGHT_DIR)
        parser.add_argument('--save_weight_dir', type=str, help='directory of saving model weights',
                            default=SAVE_WEIGHT_DIR)
        parser.add_argument('--log_dir', type=str, help='directory of saving logs', default=SAVE_LOG_DIR)
        parser.add_argument('--train_data_dir', type=str, help='directory of training data', default=TRAIN_DATA_DIR)
        #parser.add_argument('--test_data_dir', type=str, help='directory of testing data', default=TEST_DATA_DIR)
        # parser.add_argument('--gpu', type=str, help='gpu device', default='0')

        opts = parser.parse_args()
        return opts

    args = get_args()
    config = {}
    config['pre_step'] = 1
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



    pred_step = 1
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    os.makedirs(args.save_weight_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(join(args.log_dir, 'SR_Stage_4%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Initialize the model
    print('Initializing model...')
    gen = build_model(config).cuda()
    #gen = Generator7(pred_step, device=0).cuda(0)
    dis = MsImageDis().cuda()
    #fake_pool = ImagePool(500)
    #real_pool = ImagePool(500)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr / 2, betas=(0, 0.9), weight_decay=1e-4)
    opt_dis = optim.Adam(dis.parameters(), lr=args.lr * 2, betas=(0, 0.9), weight_decay=1e-4)

    # Load pre-trained weight
    if args.load_pretrain:
        print('Loading model weight...at epoch 140')
        gen.load_state_dict(torch.load(join(args.load_weight_dir, 'Gen_former_500')))
        dis.load_state_dict(torch.load(join(args.load_weight_dir, 'Dis_former_500')))

    # Load data
    print('Loading data...')
    #transformations = transforms.Compose([ToTensor(), Normalize(mean, std)])
    transformations = transforms.Compose([Resize(192), CenterCrop(192), ToTensor(), Normalize(mean, std)])
    #train_data = torch.load(args.train_data_dir)
    #train_dataset = TensorDataset(train_data['gt'], train_data['mask'], train_data['iner'])
    #train_dataset = TensorDataset(train_data['gt'], train_data['mask'])
    #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_data = dataset_norm(root=args.train_data_dir, transforms=transformations, imgSize=192, inputsize=128)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    print('train data: %d images' % (len(train_loader.dataset)))
    # if args.test_flag:
    #     test_data = dataset_around(root=args.test_data_dir, transforms=transformations, crop='center', imgSize=128)
    #     test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
    #     print('test data: %d images' % (len(test_loader.dataset)))

    # Train & test the model
    for epoch in range(140, 1 + args.epochs):
        print("----Start training[%d]----" % epoch)
        #train(gen, dis, fake_pool, real_pool, opt_gen, opt_dis, epoch, train_loader, writer)
        train(gen, dis, opt_gen, opt_dis, epoch, train_loader, writer)
        # if args.test_flag:
        #     print("----Start testing[%d]----" % epoch)
        #     test(gen, dis, epoch, test_loader, writer)

        # Save the model weight
        if (epoch % 20) == 0:
            torch.save(gen.state_dict(), join(args.save_weight_dir, 'Gen_former_%d' % epoch))
            torch.save(dis.state_dict(), join(args.save_weight_dir, 'Dis_former_%d' % epoch))

    writer.close()
