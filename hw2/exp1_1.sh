#!/bin/bash

for K in 64
do
	for L in 2 4 8 16
	do
		name="exp1_1"
		/bin/bash ./py-sbatch.sh -m hw2.experiments run-exp -n $name -K $K -L $L -P 2 -H 1000 500 100   
	done
done
