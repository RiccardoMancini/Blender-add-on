#!/bin/bash

#conda create -n gdna2 python=3.10 -y
#conda activate gdna2
#conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
#conda install cudatoolkit -y
#conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
#conda install pytorch3d -c pytorch3d -y
#conda env update --file env.yml

#cd gdna || { echo "Failure on dir gdna!"; exit 1; }
#python setup.py install
#conda list


while getopts c:b: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        c) CONDA_PATH=${OPTARG};;
        b) BLENDER_PATH=${OPTARG};;
    esac
done

cd $BLENDER_PATH || { echo "Failure on dir ${BLENDER_PATH}!"; exit 1; }
mv python _python
ls
#sudo ln -s $CONDA_PATH python || { echo "Failure on dir ${CONDA_PATH}!"; exit 1; }


unset CONDA_PATH
unset BLENDER_PATH

# source gdna/download_data.sh
