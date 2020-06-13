#!/bin/bash

dst=$1

wget --content-disposition  https://zenodo.org/record/439811/files/Train-B_batch{1,2}.tbz2?download=1

mkdir -p $dst

cat Train-B_batch{1,2}.tbz2 | tar -jxvf - -i --strip-components=2 -C $dst

python `dirname "$0"`/ich.py $dst