import os
# from cv2 import transform
import numpy as np
import warnings
import torch
import pytorch_lightning as pl
from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, CIFAR10DataModule, ImagenetDataModule
from pytorch_lightning.loggers import WandbLogger
from custom_transforms import CustomTransforms

from simmim import *
import flags_mag
from agglomerator import Agglomerator
from utils import apply_transforms

from pytorch_lightning.callbacks import ModelSummary

from absl import app
from absl import flags
FLAGS = flags.FLAGS

def init_all():
    warnings.filterwarnings("ignore")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # torch.backends.cudnn.deterministic = True

    pl.seed_everything(FLAGS.seed)
    torch.cuda.empty_cache()
    
    FLAGS.batch_size //= torch.cuda.device_count()
    FLAGS.resume_training = True if FLAGS.supervise else FLAGS.resume_training
    FLAGS.max_epochs += FLAGS.max_epochs_finetune if FLAGS.supervise else 0 #FLAGS.max_epochs


def main(argv):
    init_all()
    wandb_logger = WandbLogger(project="MAg", name=FLAGS.exp_name)

    DataModuleWrapper = {
        "MNIST": MNISTDataModule,
        "FashionMNIST": FashionMNISTDataModule,
        "CIFAR10": CIFAR10DataModule,
        "IMAGENET": ImagenetDataModule
    }

    dm = DataModuleWrapper[FLAGS.dataset](
        "./datasets", 
        batch_size=FLAGS.batch_size, 
        shuffle=True, 
        pin_memory=True, 
        drop_last=True
    )

    # apply_transforms(dm, FLAGS)
    ct = CustomTransforms(FLAGS)

    # Apply trainsforms

    dm.train_transforms = ct.train_transforms[FLAGS.dataset]
    dm.val_transforms = ct.test_transforms[FLAGS.dataset]
    dm.test_transforms = ct.test_transforms[FLAGS.dataset]

    encoder = Agglomerator(FLAGS)
    
    model = SimMIM(
        FLAGS = FLAGS,
        encoder = encoder
    )

    wandb_logger.watch(model, log='all')

    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.load_checkpoint_dir)
    

    trainer = pl.Trainer(
        gpus=-1, 
        resume_from_checkpoint=checkpoint_dir if FLAGS.resume_training else None, 
        strategy='ddp',
        max_epochs=FLAGS.max_epochs, 
        logger=wandb_logger,
        precision="bf16",
        track_grad_norm=2,
        detect_anomaly=True,
        callbacks=[ModelSummary(max_depth=-1)],
        gradient_clip_val=0.5
    )

    trainer.fit(model, dm)


if __name__ == '__main__':
    app.run(main)