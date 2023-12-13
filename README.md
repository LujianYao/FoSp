# FoSp: Focus and Separation Network for Early Smoke Segmentation
This repository is the implementation of our paper: FoSp: Focus and Separation Network for Early Smoke Segmentation. We are organizing the code and it will be open-sourced soon.

## Overview
Concentrating on the formulation of smoke images, we first introduce a Focus module employing bidirectional cascade which guides low-resolution and high-resolution features towards mid-resolution to locate and determine the scope of smoke, reducing the miss detection rate. Next, we propose a Separation module that separates smoke images into a pure smoke foreground and a smoke-free background, enhancing the contrast between smoke and background fundamentally, improving segmentation precision. Finally, a Domain Fusion module is developed to integrate the distinctive features of the two modules which can balance recall and precision to achieve high F_beta. The proposed model pipeline is: 
![Pipeline](imgs/FoSp_pipeline.png)


## Visualization
It can be observed that our FoSp can obtain finer foreground for both small and transparent smoke.
![Visualization](imgs/FoSp_prediction.png)


