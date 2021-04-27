#!/bin/sh

pip install transformers
pip install adabelief-pytorch==0.2.0
pip install ranger-adabelief==0.1.0

#!pip install torchtext#==0.8.1
#==0.7.0
pip install torchaudio
#==1.2.2
pip install pytorch-lightning
#==3.4.0

conda remove wrapt

pip install comet-ml

#git clone https://github.com/speechbrain/speechbrain.git
#cd speechbrain
#pip install -r requirements.txt
#pip install --editable .
#cd ..