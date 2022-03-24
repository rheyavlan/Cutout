#!/bin/bash
#SBATCH --job-name=resnet_gpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --output=./resnet_gpu_32.out
#SBATCH --error=./resnet_gpu_32.err
#SBATCH --gres=gpu:1
singularity exec  --nv --overlay $SCRATCH/pytorch-example/overlay-7.5GB-300K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; 
python train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16"
