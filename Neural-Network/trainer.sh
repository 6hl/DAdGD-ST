#!/bin/bash
for ((s=0; a <= 1; a++))
do
  for ((a=0; a <= 6; a++))
  do
    for ((b=1; b <= 3; b++))
    do
      if [ $s -eq 0 ]; then
        python train.py -t $a -r $b -s
      else
        python train.py -t $a -r $b
      fi
    done
  done
done
