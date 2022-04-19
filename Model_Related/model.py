import csv
import shutil
import numpy as np
# from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from monai.networks import normal_init
from monai.networks.layers import Norm
from monai.networks.nets import UNet
import monai.optimizers
from monai.utils import set_determinism
from monai.transforms import (
    CropForeground, SpatialCrop
)

from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, ConfusionMatrixMetric
from monai.losses import DiceLoss, TverskyLoss, GeneralizedDiceLoss, DiceCELoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from pytorch_lightning.utilities.seed import seed_everything
from monai.data import list_data_collate, decollate_batch, write_nifti, ThreadDataLoader, Dataset, CacheDataset, \
    DataLoader

from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from monai.data.image_reader import NibabelReader

import torch
import pytorch_lightning as pl
import glob
import os
from Data_Related import Get_TrTs, Split_TVT
from .process import train_trans, val_transforms, test_transforms, post_pred, post_label, draw_transform, \
    KeepLargestConnectedComponent


class basemodule(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        if model is None:
            Warning("No model is passed in, using UNet to continue")
            self._model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=4,
                norm=Norm.INSTANCE,
            )

        else:
            self._model = model

        # 网络初始化
        self._model.apply(normal_init)

        # 损失函数定义 二分类问题 用sigmiod 激活
        self.def_loss_function(args['loss_function_type'])

        # 二分类情况 的各种指标
        self.def_metric()

        # 验证集指标初始化
        self.best_val_dice = 0
        self.best_val_epoch = 0

        # 传入的超参在这里设置
        self.batch_size = args['batch_size']
        self.learning_rate = args['learning_rate']
        self.roi_size = args['roi_size']
        self.sw_batch_size = args['sw_batch_size']
        self.data_dir = args['data_dir']
        self.optimizer_type = args['optimizer_type']
        self.classes = args['classes']
        self.seed = args['seed']
        self.two_stage = args['two_stage']

        # 复现种子点
        set_determinism(seed=args['seed'])
        seed_everything(seed=args['seed'])

        # nii.文件读取对象，需要用这个对象读取原图的方向矩阵
        self.reader = NibabelReader()
        self.save_hyperparameters(ignore=['model', 'test_transforms', 'train_transforms',
                                          'val_transforms', 'post_pred_transform',
                                          'post_label_transform',
                                          'draw_transform',
                                          'KeepLargestConnectedComponent'
                                          ])

    def def_metric(self):

        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False)
        self.hd_metric = HausdorffDistanceMetric(
            include_background=False, reduction="mean", get_not_nans=False)
        self.sd_metric = SurfaceDistanceMetric(
            include_background=False, reduction="mean", get_not_nans=False)
        # precision recall 这个混淆矩阵可以写在一起，分开写精确一点，知道谁是谁
        self.precision_metric = ConfusionMatrixMetric(metric_name="precision", include_background=False,
                                                      reduction="mean", get_not_nans=False, compute_sample=True)
        self.recall_metric = ConfusionMatrixMetric(metric_name="recall", include_background=False, reduction="mean",
                                                   get_not_nans=False, compute_sample=True)

    def def_loss_function(self, loss_function_type):
        if loss_function_type == "DiceLoss":
            self.loss_function = DiceLoss(
                to_onehot_y=False, sigmoid=True, include_background=False)
        elif loss_function_type == "TverskyLoss":
            self.loss_function = TverskyLoss(
                to_onehot_y=False, sigmoid=True, include_background=False)
        elif loss_function_type == "GeneralizedDiceLoss":
            self.loss_function = GeneralizedDiceLoss(
                to_onehot_y=False, sigmoid=True, include_background=False)
        elif loss_function_type == "DiceFocalLoss":
            self.loss_function = DiceFocalLoss(
                to_onehot_y=False, sigmoid=True, include_background=False)
        elif loss_function_type == "DiceCELoss":
            self.loss_function = DiceCELoss(
                to_onehot_y=False, sigmoid=True, include_background=False)

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # train_dict = Split_TrTs(self.data_dir, 'Tr')
        # val_dict = Split_TrTs(self.data_dir, 'Ts')
        # test_dict = val_dict
        train_dict, val_dict, test_dict = Split_TVT(self.data_dir, 0.7, 0.2)

        # 确定transform
        train_transforms = train_trans(self.roi_size)
        # 复现局部种子点
        train_transforms.set_random_state(seed=self.seed)
        val_transforms.set_random_state(seed=self.seed)
        test_transforms.set_random_state(seed=self.seed)

        self.train_ds = Dataset(data=train_dict, transform=train_transforms)
        self.val_ds = Dataset(data=val_dict, transform=val_transforms)
        self.test_ds = Dataset(data=test_dict, transform=test_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, num_workers=2,
                                  batch_size=self.batch_size,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds, batch_size=1, num_workers=2)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_ds, batch_size=1, num_workers=2)
        return test_loader

    # 定义优化器
    def configure_optimizers(self):
        # Novograd optimizer
        if self.optimizer_type == "Novograd":
            self.optimizer = monai.optimizers.Novograd(
                self._model.parameters(), self.learning_rate)
        elif self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self._model.parameters(), self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda x: 1.0)

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
        }

    def training_step(self, batch, batch_idx):
        # loss.backward(),optimizer.step()之类的操作 lightning将其屏蔽
        images, labels = batch["image"], batch["label"]

        output = self.forward(images)
        train_loss = self.loss_function(output, labels)

        # 调用self.log记录指标
        self.log("train_loss", train_loss, logger=True, prog_bar=False)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, logger=True)
        return {"loss": train_loss, "lr": self.optimizers().param_groups[0]['lr']}

    def on_validation_epoch_start(self):
        # CSV HEAD
        # 在每个验证epoch开始的时候，写入相关信息，以便区别不同epoch下的记录
        self.out_total_csv_path = os.path.join(
            self.logger.log_dir, 'total_validation_result.csv')
        out_content = ["epcoh:" + str(self.current_epoch) + "validation"]
        self.write_csv(self.out_total_csv_path,
                       out_content, mul=False, mod='a+')
        out_content = ["name"]
        for i in range(0, self.classes):
            out_content.extend(["class_" + str(i)])
        out_content.extend(
            ["Hausdorff_distance", "SurfaceDistanceMetric", "precision", "recall"])

        self.write_csv(self.out_total_csv_path,
                       out_content, mul=False, mod='a+')

    def write_result(self, file_name, outputs_for_dice, labels):
        """

        Args:
            file_name: 当前滑动窗推理的数据
            outputs_for_dice:推理的结果
            labels: 标签

        Returns:

        """
        file_name = file_name
        self.recall_metric(y_pred=outputs_for_dice, y=labels)
        self.precision_metric(y_pred=outputs_for_dice, y=labels)
        dice = self.dice_metric(y_pred=outputs_for_dice, y=labels).item()
        hd = self.hd_metric(y_pred=outputs_for_dice, y=labels).item()
        sd = self.sd_metric(y_pred=outputs_for_dice, y=labels).item()
        precision = self.precision_metric.aggregate()[0].item()
        recall = self.recall_metric.aggregate()[0].item()
        print('\n', file_name)
        print("Dice", dice)
        print("Hausdorff_distance ", hd)
        print("SurfaceDistance", sd)
        print("precision", precision)
        print("recall", recall)
        out_content = [file_name]
        for i in range(0, self.classes):
            out_content.append(dice)
        out_content.extend([hd, sd, precision, recall])
        self.write_csv(self.out_total_csv_path,
                       out_content, mul=False, mod='a+')

    # 验证步骤
    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.roi_size
        sw_batch_size = self.sw_batch_size
        # monai 滑动窗推理
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward, mode="gaussian")
        loss = self.loss_function(outputs, labels)
        outputs_for_dice = [post_pred(i)
                            for i in decollate_batch(outputs)]
        labels = [post_label(i) for i in decollate_batch(labels)]

        # 写入数据
        self.write_result(os.path.basename(
            batch['image_meta_dict']['filename_or_obj'][0]), outputs_for_dice, labels)

        outputs = [draw_transform(i) for i in decollate_batch(outputs)]
        # 画图对比验证
        plot_2d_or_3d_image(tag="result", data=outputs, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="image", data=images, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)
        plot_2d_or_3d_image(tag="label", data=labels, step=self.current_epoch, writer=self.logger.experiment,
                            frame_dim=-1)

        return {"val_loss": loss, "val_number": len(outputs)}

    # 验证结束 更新指标
    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.hd_metric.reset()
        self.sd_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()

        mean_val_loss = torch.tensor(val_loss / num_items)

        self.log("mean_val_dice", mean_val_dice, prog_bar=True)
        self.log("mean_val_loss", mean_val_loss, prog_bar=True)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print('\n',
              f"current epoch: {self.current_epoch} "
              f"current mean dice: {mean_val_dice:.4f}"
              f"\nbest mean dice: {self.best_val_dice:.4f} "
              f"at epoch: {self.best_val_epoch}"
              )

        return

    def write_csv(self, csv_name, content, mul=True, mod="w"):
        """write list to .csv file."""

        with open(csv_name, mod) as myfile:
            writer = csv.writer(myfile)
            if mul:
                writer.writerows(content)
            else:
                writer.writerow(content)

    def on_test_epoch_start(self) -> None:
        # CSV HEAD
        #  在测试 epoch开始的时候，写入相关信息，以便区别不同epoch下的记录
        self.out_total_csv_path = os.path.join(
            self.logger.log_dir, 'total_seg_result.csv')
        out_content = ["name"]
        for i in range(0, self.classes):
            out_content.extend(["class_" + str(i)])
        out_content.extend(
            ["Hausdorff_distance", "SurfaceDistanceMetric", "precision", "recall"])
        self.write_csv(self.out_total_csv_path,
                       out_content, mul=False, mod='a+')

    # 测试test_step
    def test_step(self, batch, batch_idx):

        images, labels = batch["image"], batch["label"]
        file_name = os.path.basename(
            batch['image_meta_dict']['filename_or_obj'][0])
        roi_size = self.roi_size
        sw_batch_size = self.sw_batch_size
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward, mode="gaussian"
        )
        if self.two_stage:
            outputs = post_pred(outputs)
            outputs = KeepLargestConnectedComponent(outputs)

            result = CropForeground(
                select_fn=lambda x: x > 0, return_coords=True, margin=10)(outputs)

            # 需要获得 H W D 的最大最小
            bbox = np.array([[result[1][0], result[2][0]],  # min H max H
                             [result[1][1], result[2][1]],  # min W max W
                             [result[1][2], result[2][2]]])  # min D max D

            stage2_image = SpatialCrop(roi_start=[bbox[0][0], bbox[1][0], bbox[2][0]],
                                       roi_end=[bbox[0][1], bbox[1][1], bbox[2][1]])(images)

            stage2_predimag = sliding_window_inference(
                stage2_image, roi_size, sw_batch_size, self.forward, mode="gaussian"
            )
            stage2_predimag = post_pred(stage2_predimag)

            outputs = torch.zeros(stage2_predimag.shape[0], images.shape[1], images.shape[2], images.shape[3],
                                  images.shape[4]).cuda()

            for i in range(0, stage2_predimag.shape[0]):
                print(i)
                outputs[i][bbox[0][0]:bbox[0][1],
                bbox[1][0]:bbox[1][1],
                bbox[2][0]:bbox[2][1]] = stage2_predimag[i]

            outputs_for_dice = outputs

        else:
            outputs_for_dice = [post_pred(i) for i in outputs]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        draw = [draw_transform(i) for i in decollate_batch(outputs)]
        # 写入结果
        self.write_result(os.path.basename(
            batch['image_meta_dict']['filename_or_obj'][0]), outputs_for_dice, labels)
        data_path = batch['image_meta_dict']['filename_or_obj'][0]
        data = self.reader.read(data_path)
        _, _meta = self.reader.get_data(data)

        # 将测试结果写入到硬盘方便对比
        if not os.path.exists(os.path.join(self.logger.log_dir, "predict")):
            os.makedirs(os.path.join(self.logger.log_dir, "predict"))

        write_nifti(data=draw[0].squeeze(0), file_name=os.path.join(self.logger.log_dir, "predict", file_name),
                    output_dtype=np.int16, resample=False,
                    output_spatial_shape=_meta['spatial_shape'],
                    affine=_meta['affine'])
        torch.cuda.empty_cache()

    # 测试结束
    def test_epoch_end(self, outputs):
        # 重设 monai 相关指标
        mean_test_dice = self.dice_metric.aggregate().item()
        print("\ntest_set_mean dice", mean_test_dice)
        self.dice_metric.reset()
        self.hd_metric.reset()
        self.sd_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
