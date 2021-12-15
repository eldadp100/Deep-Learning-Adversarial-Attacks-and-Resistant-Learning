# Deep Learning Adversarial Attacks and Resistant Learning
based on Towards Deep Learning Models Resistant to Adversarial Attacks paper - https://arxiv.org/pdf/1706.06083.pdf.

For any grading problem please send me an email to eldadperetz@mail.tau.ac.il.

The pdf / word report called final_proj.pdf / final_proj.docx. 

## General information
The project consists of 2 parts:
1. Theoretical background to adversarial attacks, detailed summary of the paper, review on related papers and experiments that verify the results on GTSRB (German Traffic Sign Recognition) dataset.
2. The implemetation in pytorch on GTSRB case study.

The dataset of GTSRB is taken from https://github.com/tomlawrenceuk/GTSRB-Dataloader.

## How to run the project
(the exact commands are for GTSRB dataset, for MNIST see the comment below)
1. clone this repository (the dataset also included).
2. If you run on GPU set a specific GPU using: export CUDA_VISIBLE_DEVICES=YOUR_GPU_NUMBER
3. run from the project directory: "python experiments.py --dataset-name traffic_signs"

for MNIST - "python experiments.py --dataset-name MNIST".

see configs.py to set configs. The configurations on this reposetory are tested and match the to the report.  

## Requirements
1. Dataset - please download the dataset from this repo. (clone this repo including data folder)
2. Conda Environment - I used hw4_env to execute the project. (Activate using "conda activate hw4_env") 
3. Libraries: Pytorch, Torchvision and some other known. All apear in hw4_env.
4. Both GPU and CPU supported. GPU is more recommnended.
## Notes
In nova I occurred a bug "ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by home/wolf/sagieb/course/miniconda3/envs/hw4_env/lib/python3.7/site-packages/kiwisolver.cpython-37m-x86_64-linux-gnu.so)".
To fix the bug I add: "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wolf/sagieb/course/miniconda3/lib/" 

## Execution Example
see an_example_of_execution_results folder README.md.

## The files in the project:
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

