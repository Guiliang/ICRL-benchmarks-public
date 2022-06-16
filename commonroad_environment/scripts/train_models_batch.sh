#!/usr/bin/env bash

for i in $(seq 1 10)
do
  for m in "default" "sparse" "dense"
  do
    nohup bash train_models.sh $i $m > train_models_${i}_${m}.out &
  done
  sleep 10
done
