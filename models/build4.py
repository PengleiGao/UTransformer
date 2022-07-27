from .swin_transformer4 import SwinTransformer4
#from .trans_ablation import Ablation
import torch
import random

def build_model(config):
    model_type = config['TYPE']
    if model_type == 'swin':
        model = SwinTransformer4(pre_step=config['pre_step'],
                                img_size=config['IMG_SIZE'],
                                patch_size=config['SWIN.PATCH_SIZE'],
                                in_chans=config['SWIN.IN_CHANS'],
                                embed_dim=config['SWIN.EMBED_DIM'],
                                depths=config['SWIN.DEPTHS'],
                                num_heads=config['SWIN.NUM_HEADS'],
                                window_size=config['SWIN.WINDOW_SIZE'],
                                mlp_ratio=config['SWIN.MLP_RATIO'],
                                qkv_bias=config['SWIN.QKV_BIAS'],
                                qk_scale=config['SWIN.QK_SCALE'],
                                drop_rate=config['DROP_RATE'],
                                drop_path_rate=config['DROP_PATH_RATE'],
                                ape=False,
                                patch_norm=config['SWIN.PATCH_NORM'],
                                use_checkpoint=config['TRAIN.USE_CHECKPOINT'])
    elif model_type == 'ablation':
        model = Ablation(pre_step=config['pre_step'],
                         img_size=config['IMG_SIZE'],
                         patch_size=config['SWIN.PATCH_SIZE'],
                         in_chans=config['SWIN.IN_CHANS'],
                         embed_dim=config['SWIN.EMBED_DIM'],
                         depths=config['SWIN.DEPTHS'],
                         num_heads=config['SWIN.NUM_HEADS'],
                         window_size=config['SWIN.WINDOW_SIZE'],
                         mlp_ratio=config['SWIN.MLP_RATIO'],
                         qkv_bias=config['SWIN.QKV_BIAS'],
                         qk_scale=config['SWIN.QK_SCALE'],
                         drop_rate=config['DROP_RATE'],
                         drop_path_rate=config['DROP_PATH_RATE'],
                         ape=False,
                         patch_norm=config['SWIN.PATCH_NORM'],
                         use_checkpoint=config['TRAIN.USE_CHECKPOINT'])
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
