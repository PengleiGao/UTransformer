import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.utils import *
from DiffAugment import *

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self):
        super(MsImageDis, self).__init__()
        self.gan_type = 'ralsgan'
        self.use_r1 = True
        self.dim = 64
        self.num_scales = 3
        self.n_layer = 4
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [nn.Conv2d(3, dim, (4, 4), (2, 2), (1, 1)), nn.LeakyReLU(0.2, True)]
        for i in range(self.n_layer - 1):
            cnn_x += [nn.Conv2d(dim, min(2*dim,self.dim*8), (4, 4), (2, 2), (1, 1)), nn.InstanceNorm2d(min(2*dim,self.dim*8),affine=True), nn.LeakyReLU(0.2, True)]
            dim =min(2*dim,self.dim*8)
        cnn_x += [nn.Conv2d(dim , 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        #x = DiffAugment(x)
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        input_real.requires_grad_()
        outs0 = self.forward(input_fake)
        outs1= self.forward(input_real)
        loss =0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) + F.binary_cross_entropy(F.sigmoid(out1), all1))
            elif self.gan_type == 'ralsgan':
                loss += torch.mean((out1 - torch.mean(out0) - 1) ** 2) + torch.mean((out0 - torch.mean(out1) + 1) ** 2)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
            if self.use_r1:
                loss+= self.r1_reg(out1, input_real)
        return loss

    def calc_gen_loss(self, input_fake, input_real):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        outs1= self.forward(input_real)
        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'ralsgan':
                loss += torch.mean((out0 - torch.mean(out1) - 1) ** 2) + torch.mean((out1 - torch.mean(out0) + 1) ** 2)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def r1_reg(self, d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg
