#!/bin/bash

set -e
set -u

dumpDir=$1
sets="test dev train"
tagFile="tag.txt"
targetFile="weight.txt sentence.txt sentence2.txt"

for s in $sets; do
    echo $s
    for f in $targetFile; do
        echo ${dumpDir}/$s/$f
        python ../appendTag.py $dumpDir/$s/$tagFile $dumpDir/$s/$f $dumpDir/$s/$f.tmp
        python ../mapOrder.py ../../MAP_scripts/CQA-${s}.xml.subtaskB.relevancy $dumpDir/$s/$f.tmp $dumpDir/$s/$f.tmp2
        python ../rmTag.py $dumpDir/$s/$f.tmp2 $dumpDir/$s/$f.sorted
        rm -rf $dumpDir/$s/$f.{tmp,tmp2}
    done
done
