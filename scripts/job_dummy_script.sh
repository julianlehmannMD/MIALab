#!/bin/bash

#module load Anaconda3
eval "$(conda shell.bash hook)"

# Load your environment
conda activate mialab

while read -r line
do 
	echo "Starts main file with the argument" $line
	python ../bin/main.py --use_filter=$line
done < filter.txt

# Run your code
#run python ../bin/main.py