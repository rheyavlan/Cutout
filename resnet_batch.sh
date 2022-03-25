#!/bin/bash
#SBATCH --job-name=resnet_gpu
#SBATCH --nodes=2
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=8GB
#SBATCH --output=out/resnet_gpu_32.out
#SBATCH --error=out/resnet_gpu_32.err
#SBATCH --gres=gpu:1
module load anaconda3/2020.07
singularity exec  --nv --overlay $SCRATCH/pytorch-example/overlay-7.5GB-300K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; 
pip install matplotlib; python train.py --batch_size 64 --epochs 200 --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16"
