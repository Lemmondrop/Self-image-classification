# Self-image-classification
---
## Overview

This project solves the problem of classfying with self-image, and in this study, we classified three species of pine trees in Republic of Korea using CNNs.


project has been published in KCI(JKAIS, Journal of Korea Academia-Industrial cooperation Society)

* __DOI: https://doi.org/10.5762/KAIS.2023.24.5.660__

The article is titled __"Image Recoginition Based Classification of Native Pine Tree Species in Korea"__

## Notice

Note that the code in this git is __not fully written, Different users will have different file paths and image filenames, so you'll need to modify those before using it__. 

## Manual

It consists of 3 ipynb files in total, and the code is python.

The order of use of the files is
- [Self-image preprocessing](https://github.com/Lemmondrop/Self-image-classification/blob/main/self-image%20preprocessing.ipynb)

- [CNN(include CAM, imagedatagenerator)](https://github.com/Lemmondrop/Self-image-classification/blob/main/CNN%20(include%20CAM%2C%20Imagedatagenerator).ipynb)

- [Labeling data(feature importance)](https://github.com/Lemmondrop/Self-image-classification/blob/main/Labeling%20data(feature%20importance).ipynb)

With self-image preprocessing, you can preprocess your own image files into an easy-to-import format and save them as a single file.

The image classification is then performed using the CNN (include CAM, imagedatagenerator) file.

The labeling data code is not mandatory.
If a labeled text file exists, you can use it to compare classification results.
