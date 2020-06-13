#!/bin/bash

dst=$3

wget --user=$1 --password=$2 http://www.fki.inf.unibe.ch/DBs/iamDB/data/{forms/{formsA-D,formsE-H,formsI-Z}.tgz,xml/xml.tgz}

mkdir -p $dst/{forms,xml,pargs}

cat {formsA-D,formsE-H,formsI-Z}.tgz | tar -zxvf - -i -C $dst/forms
tar zxvf xml.tgz -C $dst/xml

python `dirname "$0"`/iam_par_gt.py $dst