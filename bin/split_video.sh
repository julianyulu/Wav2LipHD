#!/bin/bash
set -ue

inVideo=$1
step=10 # seconds per video seg
filename=$(basename $inVideo)
subfolder="${filename/.mp4/_splits}"
outDir=$2/$subfolder

echo Input video: $inVideo
echo Output dir: $outDir
mkdir -p $outDir || true

durStr=$(ffmpeg -i $inVideo 2>&1|grep Duration | cut -d , -f 1)
echo $durStr
h=$(echo $durStr|cut -d : -f 2)
m=$(echo $durStr|cut -d : -f 3)
s=$(echo $durStr|cut -d : -f 4)
secs=$(echo "$h *3600 + $m * 60 + $s"|bc)
echo seconds, $secs

secs=$(echo "$secs - $step"|bc)
for i in $(seq 0 $step $secs);do
    head=$i
    tail=$(echo "$i + $step"|bc)
    echo sec from - to: $head $tail
    ffmpeg -loglevel panic -threads 1 -ss $head -to $tail -i $inVideo -filter:v fps=25 -ac 1 -ar 16000 $outDir/sec_${head}_${tail}.mp4
done
