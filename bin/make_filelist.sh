#!/bin/bash

# indir='./preprocessed_cn'
# outdir='./filelists'

indir=$1 # e.g. 'preprocessed_data/ZYDH6_pad50'
outdir=$2

mkdir -p $outdir
for subdir in $(ls $indir);do
    outfile=$outdir/"$subdir.txt"
    echo Writing $outfile ...
    touch $outfile

    for fname in $(ls $indir/$subdir);do
	echo "$subdir/$fname" >> $outfile
    done;

done;

    
