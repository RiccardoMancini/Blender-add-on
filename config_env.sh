#!/bin/bash

if conda info --envs | grep -w gdna;
  then
    echo "gdna conda env already exists!";
  else
    conda create -n gdna python=3.10 -y
    conda activate gdna
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
    conda install cudatoolkit -y
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
    conda install pytorch3d -c pytorch3d -y
    conda env update --file env.yml

    cd gdna || { echo "Failure on dir gdna!"; return; }
    python setup.py install
    conda list
    echo "gdna conda env created succesfully!"
    cd ..
fi

if [ "$#" -lt 4 ]
  then
    echo "Incorrect number of arguments."
    return
  else
    while getopts c:b: flag
      do
          # shellcheck disable=SC2220
          case "${flag}" in
              c) CONDA_PATH=${OPTARG};;
              b) BLENDER_PATH=${OPTARG};;
          esac
      done
    if test -d $BLENDER_PATH; then
      echo "${BLENDER_PATH} directory exists.";
      else
        echo "${BLENDER_PATH} directory not exists.";
        return
    fi
    if test -d $CONDA_PATH; then
      echo "${CONDA_PATH} directory exists.";
      else
        echo "${CONDA_PATH} directory not exists.";
        return
    fi
fi

cd $BLENDER_PATH || { echo "Failure on dir ${BLENDER_PATH}!"; return; }
if test -d _python; then
    echo "Conda env already linked to Blender.";
  else
    mv python _python
    sudo ln -s $CONDA_PATH python || { echo "Failure on dir ${CONDA_PATH}!"; return; } | exit
    echo "Conda env linked to Blender.";
fi
cd -

if test -d gdna; then
  cd gdna || { echo "Failure on dir gdna!"; return; }
fi

if test -d data && test -d outputs; then
    echo "Model weights already downloaded.";
  else
    rm -r data
    rm -r outputs
    source download_data.sh
    echo "Model weights downloaded!";
fi
cd ..

echo "Updating of model weights:";
python -m pytorch_lightning.utilities.upgrade_checkpoint gdna/outputs/renderpeople/checkpoints/last.ckpt
python -m pytorch_lightning.utilities.upgrade_checkpoint gdna/outputs/thuman/checkpoints/last.ckpt
