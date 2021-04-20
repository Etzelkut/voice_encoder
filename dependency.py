from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
#from pytorch_model_summary import summary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

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