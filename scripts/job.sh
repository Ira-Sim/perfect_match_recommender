#!/bin/bash
#SBATCH --job-name=hackbay
#SBATCH --gres=gpu:32gb:1             # Number of GPUs (per node)
#SBATCH --mem=100G               # memory (per node)
#SBATCH --time=10:5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/hackbay/slurmerror.txt
#SBATCH --output=/home/mila/c/chris.emezue/hackbay/slurmoutput.txt

###########cluster information above this line


###load environment 

module load python/3
module load cuda/11.1


source ~/scratch/hackbay-env/bin/activate


HACKBAY_DIR=~/scratch/hackbay
mkdir -p $HACKBAY_DIR 
cd $HACKBAY_DIR

python ~/hackbay/model.py 
