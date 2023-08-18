# Progetto CG
## Task
Realizzare un componente aggiuntivo per Blender che integri la rete neurale gDNA per generare avatar 3D in diverse pose e con diversi abiti.

## Setup environment
Place in 'gdna' directory and exec these commands from terminal:

```
conda create -n gdna python=3.10 -y
conda activate gdna
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install cudatoolkit -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install pytorch3d -c pytorch3d -y
conda env update --file env.yml
python setup.py install
```

## Setup environment 2.0
TODO: mettere env.yml nella root dir del progetto e testare se la rete funziona anche senza setup.py
```
source config_env.sh
```
