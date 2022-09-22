# PyTorch implementation of "SPD-CNN: A Plain CNN-Based Model Using the Symmetric Positive Definite Matrices for Cross-Subject EEG Classification with Meta-Transfer-Learning"
[![Python](https://img.shields.io/badge/python-3.7-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.4.0-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

This repository contains the PyTorch implementation for the Paper ["SPD-CNN: A Plain CNN-Based Model Using the Symmetric Positive Definite Matrices for Cross-Subject EEG Classification with Meta-Transfer-Learning"](http://journal.frontiersin.org/article/10.3389/fnbot.2022.958052/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Neurorobotics&id=958052.) 

This repository is mainly modified from the Github repository of "Meta-transfer-learning"(https://github.com/sabinechen/meta-transfer-learning).
If you have any questions on this repository or the related paper, feel free to create an issue or send me an email (Chenlezih@formail.com).


#### Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Datasets](#datasets)
* [Performance](#performance)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Introduction

To build a robust classiﬁcation algorithm for a calibration-less BCI systemcalibration-less BCI system, we propose an end-to-end model that
transforms the EEG signals into symmetric positive deﬁnite (SPD) matrices and captures the features of SPD matrices by using a CNN. To avoid the
time-consuming calibration and ensure the application of the proposed model, we use the meta-transfer-learning (MTL) method to learn the essential
features from different subjects.The main Figure are shown below.

<p align="center">
    <img src="https://www.frontiersin.org/files/Articles/958052/fnbot-16-958052-HTML-r2/image_m/fnbot-16-958052-g001.jpg" width="800"/>
</p>

> Figure1: Overall visualization of the SPD-CNN architecture. . 

<p align="center">
    <img src="https://www.frontiersin.org/files/Articles/958052/fnbot-16-958052-HTML-r2/image_m/fnbot-16-958052-g002.jpg" width="800"/>
</p>

> Figure2: Workflow of our training framework . 

<p align="center">
    <img src="https://www.frontiersin.org/files/Articles/958052/fnbot-16-958052-HTML-r2/image_m/fnbot-16-958052-g003.jpg" width="800"/>
</p>

> Figure3: Diagram of parameters variation through the learning process in different phases.

A whole visualization and full description of SPD-CNN model can be found in Figure 1.The Workflow of our training framework are shown in Figuer2 and Figuer 3.

## Getting Started
In order to run this repository, we advise you to install python 3.7 and PyTorch 1.4.0 with Anaconda.
You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and on it:
```bash
conda create --name mtl-pytorch python=3.7
conda activate mtl-pytorch
conda install pytorch=1.4.0 
```
Install other requirements(for generating the dataset and processing the data):smile::smile::
```bash
pip install parse,pillow,scipy,moabb
```
Then use the python file in the Data_generator to download and process the datasets:smile:.

## Datasets
The data that support the findings of this study are openly available in https://github.com/NeuroTechX/moabb.
## Performance 
<p align="center">
    <img src="https://www.frontiersin.org/files/Articles/958052/fnbot-16-958052-HTML-r2/image_m/fnbot-16-958052-t004.jpg" width="800"/>
</p>

## Citation

Please cite our paper if it is helpful to your work:

```bibtex
@article{chenspd,
 title={SPD-CNN: A Plain CNN-Based Model Using the Symmetric Positive Definite Matrices for Cross-Subject EEG Classification with Meta-Transfer-Learning},
  author={Chen, Lezhi and Yu, Zhuliang and Yang, Jian},
  journal={Frontiers in Neurorobotics},
  pages={168},
  publisher={Frontiers}
}
```
## Acknowledgements

Our implementations use the source code from the following repositories and users :smile::
* [Model-Agnostic Meta-Learning](https://github.com/cbfinn/maml)
* [Meta-Transfer Learning for Few-Shot Learning](https://github.com/yaoyao-liu/meta-transfer-learning#meta-transfer-learning-for-few-shot-learning)
* [EEGNet](https://github.com/aliasvishnu/EEGNet)
* [MOABB](https://github.com/NeuroTechX/moabb.)
