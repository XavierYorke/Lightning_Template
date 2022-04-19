#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Time    :   2022/4/19
# Author  :   XavierYorke
# Contact :   mzlxavier1230@gmail.com


import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary, ModelCheckpoint
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from Model_Related import basemodule
from Model_Related import net
import warnings
warnings.filterwarnings('ignore')


def pl_setup(model, log_dir, epochs, mode, resume, ckpt_path):
    tb_logger = pl.loggers.TensorBoardLogger(
        log_dir, name=cfg['name'])
    checkpoint_callback = ModelCheckpoint(
        monitor="mean_val_dice",
        filename="vessel_seg-{epoch:02d}-{mean_val_dice:.4f}",
        save_last=True,
        save_top_k=3,
        mode="max",
        save_on_train_epoch_end=True
    )
    gpu_callback = GPUStatsMonitor()
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=epochs,
        logger=tb_logger,
        checkpoint_callback=True,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        precision=16,
        callbacks=[checkpoint_callback, gpu_callback],
    )
    if mode == 'train':
        if resume is False:
            trainer.fit(model)
            trainer.test(model)
        else:
            # 恢复训练
            trainer.fit(model, ckpt_path=ckpt_path)
            trainer.test(model)
    if mode == 'test':
        trainer.test(model, ckpt_path=ckpt_path)


def main(output_dir):
    model = basemodule(model=net, args=model_cfg)
    pl_setup(model, output_dir, **pl_cfg)


if __name__ == '__main__':
    cfg = yaml.load(open('cfg.yaml'), Loader=yaml.FullLoader)
    pl_cfg = cfg['pl_cfg']
    model_cfg = cfg['model_cfg']
    main(cfg['output_dir'])
