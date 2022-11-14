#!/bin/bash
# SLURM Settings
#SBATCH --job-name="MIALab_Feature_Extraction"
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=128G
#SBATCH --mail-user=marcel.allenspach@students.unibe.ch
#SBATCH --mail-type=ALL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


module load Anaconda3
eval "$(conda shell.bash hook)"

# Load your environment
conda activate mialab

while read -r line
do 
	echo "Starts main file with the argument" $line
	srun python3 ../bin/main.py --use_filter=$line

done < filter.txt
