#!/bin/bash
cd share/data
mkdir microscopy
cd microscopy
mkdir "20170817 MS-E27 LD qDII-CLV3-DR5"
cd "20170817 MS-E27 LD qDII-CLV3-DR5"
mkdir RAW
cd RAW
wget http://flower.ens-lyon.fr/sam_patterning/data/qDII-CLV3-DR5-E27-LD-SAM7.czi
wget http://flower.ens-lyon.fr/sam_patterning/data/qDII-CLV3-DR5-E27-LD-SAM7-T5.czi
wget http://flower.ens-lyon.fr/sam_patterning/data/qDII-CLV3-DR5-E27-LD-SAM7-T10.czi
cd ..
mkdir TIF-No-organs
cd TIF-No-organs
wget http://flower.ens-lyon.fr/sam_patterning/data/qDII-CLV3-DR5-E27-LD-SAM7-No-organs.tif
wget http://flower.ens-lyon.fr/sam_patterning/data/qDII-CLV3-DR5-E27-LD-SAM7-T5-No-organs.tif
wget http://flower.ens-lyon.fr/sam_patterning/data/qDII-CLV3-DR5-E27-LD-SAM7-T10-No-organs.tif
cd ../../../../..
