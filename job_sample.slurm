#!/bin/bash
#SBATCH -A ACF-UTK0011
#SBATCH --job-name=err_sample
#SBATCH --output=output/output.o%j
#SBATCH --error=output/error.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --mem=1G
#SBATCH --gres=gpu:1

#SBATCH --qos=campus-gpu
#SBATCH --partition=campus-gpu
##SBATCH --mail-user=
##SBATCH --mail-type=ALL

./err_sample 1000000 covData/cov18.dat
./err_sample 1000000 covData/cov19.dat
./err_sample 1000000 covData/cov20.dat
./err_sample 1000000 covData/cov21.dat
./err_sample 1000000 covData/cov22.dat
./err_sample 1000000 covData/cov24.dat
