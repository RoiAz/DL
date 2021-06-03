#!/bin/bash


name="exp1_4"
/bin/bash ./py-sbatch.sh -m hw2.experiments run-exp -n $name -K 64 128 256 -L 8 -P 8 -H 1000 500 100 -M "resnet"
