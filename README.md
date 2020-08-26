# cnn_course_final
Convolutional Neural Networks Final Project
based on Towards Deep Learning Models Resistant to Adversarial Attacks paper - https://arxiv.org/pdf/1706.06083.pdf.

The project consists of 2 parts:
1. Theoretical background to adversarial attacks, detailed summary of the paper, review on related papers and experiments that verify the results on GTSRB (German Traffic Sign Recognition) dataset.
2. The implemetation in pytorch on GTSRB case study.

The dataset of GTSRB is taken from https://github.com/tomlawrenceuk/GTSRB-Dataloader.

How to run the project (on GTSRB):
1. clone this repository (the dataset also included).
2. run from the project directory: python experiments.py --dataset-name traffic_signs

for MNIST - python experiments.py --dataset-name MNIST
