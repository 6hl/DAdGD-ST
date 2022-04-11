#!/bin/bash
for ((a=0; a <= 1; a++))
do
  for ((b=1; b<= 7; b++))
  do
    python mf.py -t $b -s $a
  done
done