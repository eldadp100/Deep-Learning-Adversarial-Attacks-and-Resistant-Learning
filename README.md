# Convolutional Neural Networks Final Project
based on Towards Deep Learning Models Resistant to Adversarial Attacks paper - https://arxiv.org/pdf/1706.06083.pdf.

The project consists of 2 parts:
1. Theoretical background to adversarial attacks, detailed summary of the paper, review on related papers and experiments that verify the results on GTSRB (German Traffic Sign Recognition) dataset.
2. The implemetation in pytorch on GTSRB case study.

The dataset of GTSRB is taken from https://github.com/tomlawrenceuk/GTSRB-Dataloader.

How to run the project (on GTSRB):
1. clone this repository (the dataset also included).
2. run from the project directory: python experiments.py --dataset-name traffic_signs

for MNIST - python experiments.py --dataset-name MNIST



About the files in the project:
* experiments.py - the main file of the project. To execute the project run this file as explained in the previous section.
* attacks.py - the adversarial attacks implementations. PGD and FGSM classes are the 
specific attacks.
* helper.py - contains auxiliary tools. Hyperparameter generators and specific tasks searching
methods are implemented there. Also contains plotting method for images vector as a grid.
* logger.py - contains logger object that is used globally as it's imported in every file. It is
initialized in experiments.py.
*  models.py - here are all the networks architectures and generators are implemented. There is a
networks generator that is used in experiment 4 (capacity and robustness relation) and specific architectures for the rest of the experiments. 
* datasets.py - the GTSRB dataset is given as a folder of ppms files. GTSRB class is a Dataset type that
parse this folder. There are also Dataloaders methods.
* trainer.py - here is the training method (both typical training and adversarial training). It contains implementations to training 
management tools (Epochs, StoppingCriteria classes).
* configs.py - this file contains all the configurations to the project - both system configurations and hyperparams. 
It is imported only in experiments.py 

