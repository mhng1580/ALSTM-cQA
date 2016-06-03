#!/bin/bash

set -e
set -u
set -x

dirs=$1
#sets="test dev train"
sets="dev"
sFile="sentence.txt.sorted"
wFile="weight.txt.sorted"
figDir='fig'

for d in $dirs; do
    echo $d
    for s in $sets; do
        echo $s
        mkdir -p $d/$s/$figDir
        matlab -r "run('$d/$s/$sFile', '$d/$s/$wFile', '$d/$s/$figDir/'); quit;"
    done
done
