#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/04/19 16:18:48
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

from monai.transforms import Compose, LoadImaged, AddChanneld, Spacingd, Orientationd, ScaleIntensityRanged, \
    CropForegroundd, Resized, CopyItemsd, SpatialCropd, EnsureTyped, RandCropByPosNegLabeld, SpatialPadd, ToDeviced, \
    RandZoomd, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, RandFlipd, CastToTyped, ScaleIntensityd, \
    KeepLargestConnectedComponent, AsChannelLastd, AsChannelFirstd, \
    EnsureTyped, EnsureType, Activations, AsDiscrete


def train_trans(roi_size):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"], minv=0, maxv=1.0),
        RandCropByPosNegLabeld(keys=["image", "label"], pos=1, neg=1, num_samples=2, label_key="label",
                               spatial_size=roi_size),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transforms


val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"], minv=0, maxv=1.0),

    EnsureTyped(keys=["image", "label"])
])

test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    ScaleIntensityd(["image"], minv=0, maxv=1.0),
    EnsureTyped(keys=["image", "label"])
])

# 后处理网络输出

post_pred = Compose([EnsureType(), Activations(sigmoid=True),
                     AsDiscrete(threshold=0.5)])
# 后处理标签
post_label = Compose([EnsureType()])
# 在tensorboard画图不需要独热编码
draw_transform = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
KeepLargestConnectedComponent = KeepLargestConnectedComponent([1], independent=True, connectivity=None)
