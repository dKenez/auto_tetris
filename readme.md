# Auto Tetris - Deep Learning Project

## Setup

The following commands should be executed from the repository root

**Create virtual environment**

```
virtualenv -p python3.8 tetris
```

**Activate virtual environment**

```
source ./tetris/bin/activate
```

**Update pip**

```
pip install --upgrade pip
```

**Install from `requirements.txt`**

Note: when installing on the HPC cluster, install the Nvidia specific pytorch version

```
pip install -r requirements.txt
```

## Structure

The `src` directory contains the following

`unet.py`

Defines the unet class. Highly modifiable: layers, input channels, feature channels, and output channels can all be modified.

**TODO: Rest of the files**

The `data` directory contains the data, images and masks are separated into `images` and `masks` directories respectively. `paths_df_validated_0_to_6500_8000_to_8499.csv` contains the path to the filtered images and corresponding masks.