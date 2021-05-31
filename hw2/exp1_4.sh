#!/bin/bash


name="exp1_4"
for L in 8 16 32
do
		/bin/bash ./py-sbatch.sh -m hw2.experiments run-exp -n $name -K 32 -L $L -P 8 -H 1000 500 100 -M "resnet"
done

sleep 30

for L in 2 4 48
do
		/bin/bash ./py-sbatch.sh -m hw2.experiments run-exp -n $name -K 64 128 256 -L $L -P 8 -H 1000 500 100 -M "resnet"
done
