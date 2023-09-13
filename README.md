# Computer Graphics 2022/23 Project: gDNA for Blender (Add-On Development)
## Project Overview
Our project involves the development of a Blender add-on that seamlessly integrates the gDNA neural network. This add-on enables the generation of 3D avatars in different poses and outfits, enhancing Blender's capabilities.

## System Requirements
To use this add-on correctly, the system must meet the following requirements:
- Anaconda Distribution
- Linux OS (where the gDNA network implementation is supported)
- Up-to-date GPU drivers


## Installation Instructions
This section provides a comprehensive guide to using the add-on developed in Blender. Before proceeding, please make sure that you have downloaded this repository to your local machine.

### Setting up Conda Environment and Linking with Blender
First of all, you'll need to create an Anaconda virtual environment with all the necessary libraries, dependencies, and model weights. Next, you must establish a connection between this environment and the Blender's Python interpreter.

We have provided a convenient bash script called 'config_env.sh' that automates this entire process for you. To begin, open a terminal in the directory where you downloaded the repository and run the following command:

```source config_env.sh -c <CONDA_ENVS_PATH>/gdna -b <BLENDER_PYTHON_PATH>```

Example:
```source config_env.sh -c ~/anaconda3/envs/gdna -b ~/Blender/3.6```


### Installing the Add-on in Blender
After completing the environment setup, it's time to install the add-on in Blender. To do this, compress the entire project folder into a .zip file. Then, in Blender's 'Preferences' menu, go to the Add-ons section and install the add-on from the .zip file. You are now ready to start using the developed add-on.

### Video Tutorial
[Click here](https://mega.nz/file/5psTDIhY#i5LDguCIz5p8IDuQGTEr8ZnDW03CMYrhDmOOHdHYZPo) to watch a video tutorial for the setup.

