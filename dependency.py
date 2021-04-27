from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
#from pytorch_model_summary import summary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import pytorch_lightning as pl
from pytorch_lightning import seed_everything


from scipy.io import wavfile
import os
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import random
import torch
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model, Wav2Vec2Processor

from ge2e import GE2ELoss
import torch.nn as nn

from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

import torch.autograd as grad
import torch.nn.functional as F

import speechbrain.utils.metric_stats as ms



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def seed_e(seed_value):
  seed_everything(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value) 
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        pathh = False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.pathh = pathh

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(self.pathh, filename)
            print("we are here!!!")
            trainer.save_checkpoint(ckpt_path)