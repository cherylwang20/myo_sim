# MuJoCo XML Processing Tools

Two Python scripts for modifying MuJoCo XML files:
1. `MSK_scaling.py` - calculate the scaling factor for each body segment based on the provided mat files and the default mujoco files.
2. `auto_scaling.py`
    - create a new assets folder 'assets_test'
    - create a new file for the main xml file
    - rescale all the body mesh file size
    - rescale all the body part position and muscle insertion point
    - rescale all body segment masses based on total target mass and scaling factor


## Requirements

- Python 3.8+
- `xml.etree.ElementTree` (standard library)
- `pathlib.Path` (standard library)


### Purpose
A customized model in which users can easily scale the default 'myolegs_abdomen.xml' model based on the provided mat/mot file.
