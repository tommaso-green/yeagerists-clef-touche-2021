from argquality_dataset import *
from model import *
import wandb
from pytorch_lightning.loggers import WandbLogger
from knockknock import telegram_sender
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

CHAT_ID: int = 125901512


# @telegram_sender(token="1717011473:AAFVMLD5p_1SzCFEiPZ6Ahhr2U35ZidHGc0", chat_id=CHAT_ID)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('-hs', '--hyperparameter_search', type=bool, default=False)
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('-m', '--model_name', type=str, default='bert-base-uncased')

    parser = ArgQualityDataset.add_specific_args(parser)
    parser = ArgQualityModel.add_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    split_dict = {'train': args.train_size, 'val': args.val_size, 'test': args.test_size}
    dm = ArgQualityDataset(batch_size=args.batch_size, csv_path=args.csv_path,
                           tokenizer_name=args.model_name, split_dict=split_dict)
    dm.prepare_data()
    dm.setup()

    arg_quality_model = ArgQualityModel(model_name=args.model_name, learning_rate=args.learning_rate,
                                        weight_decay=args.weight_decay)

    if args.hyperparameter_search:
        wandb.init()
    else:
        wandb.init(project="ArgumentQuality", name=args.model_name)

    if not args.hyperparameter_search:
        my_callbacks = [ModelCheckpoint(
            monitor='val_r2',
            dirpath='model_checkpoints/',
            filename=args.model_name+'_best-{epoch:02d}-{val_r2:.2f}',
            mode='max', save_top_k=2),
            EarlyStopping(monitor='val_r2', mode='max', patience=4)]
        ckpt_flag = True #checkpoint_callback=False
    else:
        my_callbacks = []
        ckpt_flag = False

    wandb.config.update(args)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.num_epochs, val_check_interval=0.25,
                         deterministic=True, logger=wandb_logger, callbacks=my_callbacks, 
                         checkpoint_callback=ckpt_flag)
    trainer.fit(arg_quality_model, datamodule=dm)
    return wandb.run.url


if __name__ == "__main__":
    main()
