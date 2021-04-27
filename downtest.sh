#!/bin/sh

mkdir test

pip install gdown

gdown https://drive.google.com/uc?id=1eKKoKgYsfg2PqqSb-No5h1fkX-2SyHHa
gdown https://drive.google.com/uc?id=1uZAG1-LO_8nFutPmw4IEDHTmuv_vHJPK


mv veri_test.txt /test/test_division.txt

unzip -qq vox1_test_wav.zip -d test