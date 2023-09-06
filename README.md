# Progetto CG
## Task
Developing an add-on for Blender that incorporates the gDNA neural network to generate 3D avatars in various poses and outfits.

## How to install
To use the add-on:
- First of all you need to configure a Conda virtual environment with all the required libraries, dependencies, and model weights.
- Once that's done, you must link the created environment to the Blender Python interpreter.
- Finally, you can install the add-on from Blender's "Preferences" and start using it.

Below, we will provide a detailed breakdown of these steps. Before that you need to download this repository in local.

### Setup conda env

1. 

1. Download the zip from this Git and installing the add-on file inside Blender add-ons section
2. Download the weights from Pix2Vox-A
3. Insert in the add-on (Tools: Pix2Blender): the Input Image folder (3 are provided in this Git to test it), the Core folder and the Weights file.
```
source config_env.sh
```
