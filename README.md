# Computer Graphics 2022/23 project
## Overview
Developing an add-on for Blender that incorporates the gDNA neural network to generate 3D avatars in various poses and outfits.

## Requirements
- Anaconda Distribution
- Linux (OS where gDNA network was implemented)
- GPU drivers updated

## How to install
This section is a detailed guide for use developed add-on in Blender. Before that, you need to download this repository in local.

### Setup conda env and connect it with Blender
- First of all you need to set up an Anaconda virtual environment with all the required libraries, dependencies and model weights. Then you must link the created environment to the Blender Python interpreter.
A bash script 'config_env.sh' has been devoloped which automates this whole process.

- So it will be enough to open a terminal in the directory of the downloaded repository where the 'config_env.sh' file resides and run the following command:
```source config_env.sh```


### Install add-on in Blender
At this point you need to compress the entire project folder (.zip), so you can install the add-on from Blender's 'Preferences', inside Blender add-ons section, and start using it.



