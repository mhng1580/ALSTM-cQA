#!/bin/bash 

gold=$1
pred=$2

python rnn_enc/utils/mapOrder.py $1 $2 $2.tmp
python rnn_enc/MAP_scripts/ev.py $1 $2.tmp 
rm $2.tmp
