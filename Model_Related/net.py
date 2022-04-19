#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/04/19 15:46:15
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

from monai.networks.nets import UNet, SegResNetVAE, HighResNet, VNet, DynUNet
from monai.networks.layers import Norm


net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=4,
            norm=Norm.INSTANCE,
)

if __name__ == '__main__':
    print(net)
