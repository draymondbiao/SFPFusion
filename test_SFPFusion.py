"""测试融合网络"""
import argparse
import os
import random

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.msrs_data import MSRS_data
from data_loader.common import YCrCb2RGB, clamp
from data_loader.fusion_strategy import L1_Norm
from model_SFPFusion import MODEL, WaveDecoder
from torchvision import transforms
import torch
from time import time

torch.cuda.set_device(1)

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

class WCT2:
    def __init__(self, option_unpool='sum', verbose=True):
        self.verbose = verbose
 
        self.encoder = MODEL(embed_dim=[64, 128, 320, 512],  # 25M, 4.4G, 677FPS
        depths=[3, 5, 9, 3],
        num_heads=[1, 2, 5, 8],
        n_iter=[1, 1, 1, 1],
        stoken_size=[8, 4, 1, 1],

        projection=1024,

        mlp_ratio=4,
        stoken_refine=True,
        stoken_refine_attention=True,
        hard_label=False,
        rpe=False,
        qkv_bias=True,
        qk_scale=None,
        use_checkpoint=False,
        checkpoint_num=[0, 0, 0, 0],
        layerscale=[False] * 4,
        init_values=1e-6,option_unpool='sum').cuda()
        self.decoder = WaveDecoder(option_unpool).cuda()
        self.encoder.load_state_dict(torch.load('models/model_SFPFusion/1e1/Final_Encoder_epoch.model',
            map_location=torch.device('cpu')))
        self.decoder.load_state_dict(torch.load('models/model_SFPFusion/1e1/Final_Decoder_epoch.model',
            map_location=torch.device('cpu')))
        # print(self.encoder)
        # total = sum([params.nelement() for params in self.encoder.parameters()])
        # print("Number of params Encoder: {%.2f M}" % (total / 1e6))
        #
        # total = sum([params.nelement() for params in self.decoder.parameters()])
        # print("Number of params Encoder: {%.2f M}" % (total / 1e6))
        self.encoder.eval()
        self.decoder.eval()

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)



    def get_all_feature(self, ir_image,vis_y_image):
        skips = {}
        ir_skips={}
        vis_y_skips={}
        feats={'encoder': {}, 'decoder': {}}
        ir_feats = {'encoder': {}, 'decoder': {}}
        vis_y_feats = {'encoder': {}, 'decoder': {}}

        for level in [1, 2, 3, 4]:
            ir_image = self.encode(ir_image, ir_skips, level)
            vis_y_image = self.encode(vis_y_image, vis_y_skips, level)

            from torchvision import transforms
            #weighted maps generated
            # fusion = torch.sum(vis_y_image,dim=0)
            # fusion = torch.sum(fusion, dim=0)
            # fusion = fusion/vis_y_image.size()[1]
            # rgb_fused_image = transforms.ToPILImage()(fusion)
            # rgb_fused_image.save('weight/1.png')

            ir_feats['encoder'][level] = ir_image
            vis_y_feats['encoder'][level] = vis_y_image

        return skips,ir_skips,vis_y_skips,ir_feats,vis_y_feats

    def transfer(self, vis_y_image, ir_image):
       
        skips, ir_skips, vis_y_skips, ir_feats, vis_y_feats = self.get_all_feature(ir_image, vis_y_image)

        fusion_feat = {'encoder': {}, 'decoder': {}}
        fusion_skips = {}


        wct_skips=[1,2,3]

        wct2_skip_level = ['pool1', 'pool2', 'pool3']
        fusion_skips['pool1'] = [0, 0, 0]
        fusion_skips['pool2'] = [0, 0, 0]
        fusion_skips['pool3'] = [0, 0, 0]
        fusion = torch.tensor(1)
        for level in [1, 2, 3, 4]:

            fusion_feat['encoder'][level] = L1_Norm(torch.abs(vis_y_feats['encoder'][level]),
                                                                    torch.abs(ir_feats['encoder'][level]))

            if level in wct_skips:
                skip_level = wct2_skip_level[level-1]

                for component in [0, 1, 2]:  # component: [VD, HD, DD]
                    fusion_skips[skip_level][component] = (ir_skips[skip_level][component] +
                                                           vis_y_skips[skip_level][component])

        for level in [4, 3, 2, 1]:
            if level == 4:
                fusion = self.decode(fusion_feat['encoder'][level], fusion_skips, level)
            if level == 3:
                fusion = self.decode(fusion, fusion_skips, level)
            if level == 2:
                fusion = self.decode(fusion, fusion_skips, level)
            if level == 1:
                fusion = self.decode(fusion, fusion_skips, level)
        return fusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='test_data/MSRS/',
                        help='path to dataset (default: imagenet)')# 测试数据存放位置
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model', choices=['fusion_model'])
    parser.add_argument('--save_path', default='results/SFPFusion-MSRS')# 融合结果存放位置

    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    # init_seeds(args.seed)


    test_dataset = MSRS_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 如果是融合网络
    if args.arch == 'fusion_model':
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        list_no=[]
        with torch.no_grad():
            wct2 = WCT2()
            sum=0

            for vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name,image_size in test_tqdm:
                vis_image = vis_image.cuda()
                vis_y_image = vis_y_image.cuda()
                cb = vis_cb_image.cuda()
                cr = vis_cr_image.cuda()
                inf_image = inf_image.cuda()

                # try:
                start_time = time()
                fused_image = wct2.transfer(vis_y_image,inf_image)

                end_time = time()
                elapsed = end_time - start_time
                sum+=elapsed
                print(name[0],elapsed)

                fused_image = clamp(fused_image[0][0])
                # fused_image = clamp(fused_image[0][0]).cpu()
                # pred = torch.squeeze(fused_image).numpy()
                # pred_mask = np.where(fused_image > 0.5, 1, 0)
                # import imageio
                # imageio.imsave(f'{args.save_path}/{name[0]}', pred_mask[0])

            # 格式转换，因为tensor不能直接保存成图片

            # fused_image=fused_image.reshape([])
                fused_image = YCrCb2RGB(fused_image, cb[0], cr[0])
                fused_image = transforms.ToPILImage()(fused_image)
                # fused_image=fused_image.resize((size[0],size[1]),Image.BILINEAR)
                fused_image.save(f'{args.save_path}/{name[0]}')
                print(sum)
                # except:
                #     list_no.append(name[0].split('.')[0])
            # print(list_no)