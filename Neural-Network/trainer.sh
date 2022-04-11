#!/bin/bash
for ((a=0; a <= 6; a++))
do
  for ((b=1; b <= 3; b++))
  do
    python train.py -t $a -r $b -s
  done
done

for ((a=0; a <= 6; a++))
do
  for ((b=1; b <= 3; b++))
  do
    python train.py -t $a -r $b
  done
done