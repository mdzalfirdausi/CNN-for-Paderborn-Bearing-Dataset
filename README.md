# CNN for Paderborn Bearing Dataset
## Introduction
This repository is my work to implement CNN for Paderborn bearing fault dataset
Credit to: https://github.com/XiongMeijing/CWRU-1
The dataset can be downloaded from [(here)](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter)

Helper functions for data cleaning and preprocessing are written in the `helper.py` module, whereas helper functions for training using Pytorch Framework are written in the `train_helper.py` module. To select two or three layers 1D CNN to be used, please refer to `nn_model.py`

The notebook `Paderborn_Dataset.ipynb` shows the training process and the trained model is saved in the `./Model` folder.

I add the capability of data cleaning and preprocessing to select several types of sensor which is:
`force`, `phase_current_1`, `phase_current_2`, `speed`, `temp_2_bearing_module`, `torque`, `vibration_1`
