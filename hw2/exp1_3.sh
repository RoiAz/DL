#!/bin/bash

for L in 1 2 3 4
do
		name="exp1_3"
		/bin/bash ./py-sbatch.sh -m hw2.experiments run-exp -n $name -K 64 128 256 -L $L -P 8 -H 1000 500 100   
done
