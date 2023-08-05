# Progetto CG
## Task
Realizzare un componente aggiuntivo per Blender che integri la rete neurale gDNA per generare avatar 3D in diverse pose e con diversi abiti.

## Setup
Place in a directory and clone this repo:
```
git clone https://github.com/xuchen-ethz/gdna.git
cd gdna
```

Install environment:
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

Download [SMPL models](https://smpl.is.tue.mpg.de) (1.0.0 for Python 2.7 (10 shape PCs)) and move them to the corresponding locations:
```
mkdir lib/smpl/smpl_model/
mv /path/to/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_NEUTRAL.pkl
```

Download pretrained models and test motion sequences: 
```
sh ./download_data.sh
```

## Inference

Run one of the following command and check the result video in `outputs/renderpeople/video`


**"Dancinterpolation":** generate a dancing + interpolation sequence
```
python test.py expname=renderpeople +experiments=fine eval_mode=interp
```

**Disentangled Control:** change the coarse shape while keeping other factors fixed
```
python test.py expname=renderpeople +experiments=fine eval_mode=z_shape
```
To control other factors, simply change `eval_mode=[z_shape|z_detail|betas|thetas]`.

**Random Sampling:** generate samples with random poses and latent codes
```
python test.py expname=renderpeople +experiments=fine eval_mode=sample
```


**THuman2.0 Model:** run the following command with desired eval_mode for the model trained with THuman2.0
```
python test.py expname=thuman  model.norm_network.multires=6 +experiments=fine datamodule=thuman eval_mode=interp
```
Note that for this dataset we use more frequency components for the positional encoding (`model.norm_network.multires=6`) due to the rich details in this dataset. Also note that this THuman2.0 model exhibits less body shape (betas) variations bounded by the body shape variations in the training set.
