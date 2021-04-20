#!/bin/sh

mkdir data
mkdir weights

pip install gdown
gdown https://drive.google.com/uc?id=1qZ9NEJfc8AQ20ht88QX2ZlxXOoT79aMY

wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt

unzip -qq vox1_dev_wav.zip -d data