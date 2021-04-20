#!/bin/sh

mkdir data
mkdir weights

pip install gdown
gdown https://drive.google.com/uc?id=1qZ9NEJfc8AQ20ht88QX2ZlxXOoT79aMY

unzip -qq vox1_dev_wav.zip.zip -d data

wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt
pip install transformers
pip install adabelief-pytorch==0.2.0
pip install ranger-adabelief==0.1.0

#!pip install torchtext#==0.8.1
#==0.7.0
pip install torchaudio
#==1.2.2
pip install pytorch-lightning
#==3.4.0
pip install comet-ml

