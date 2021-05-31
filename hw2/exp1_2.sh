#!/bin/bash

for L in 8
do
	for K in 32 64 128 256
	do
		name="exp1_2"
		/bin/bash ./py-sbatch.sh -m hw2.experiments run-exp -n $name -K $K -L $L -P 8 -H 1000 500 100   
	done
done
