import argparse

from argquality_dataset import *
from model import *
import os
import wandb
from pytorch_lightning.loggers import WandbLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('-m', '--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--id', type=str)
    parser = ArgQualityDataset.add_specific_args(parser)
    args = parser.parse_args()


    pl.seed_everything(args.seed)
    split_dict = {'train': args.train_size, 'val': args.val_size, 'test': args.test_size}
    dm = ArgQualityDataset(batch_size=args.batch_size, csv_path=args.csv_path,
                           tokenizer_name=args.model_name, split_dict=split_dict)

    dm.prepare_data()
    dm.setup()
    ckpt_file = "model_checkpoints/" + args.ckpt
    arg_quality_model = ArgQualityModel.load_from_checkpoint(ckpt_file)

    wandb.init(project="ArgumentQuality", name=args.model_name, resume=True, id=args.id)
    args.__delattr__("id")
    args.__delattr__("ckpt")
    wandb.config.update(args)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(gpus=args.gpus, deterministic=True, logger=wandb_logger,
                         checkpoint_callback=False)

    trainer.test(arg_quality_model, datamodule=dm)
    arg_quality_model = ArgQualityModel.load_from_checkpoint(ckpt_file)


if __name__ == "__main__":
    main()