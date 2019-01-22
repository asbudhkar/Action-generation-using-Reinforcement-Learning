# Action-generation-using-Reinforcement-Learning
Experimenting with supervised RL for action generation

## Introduction
Select videos of people (a) clapping (b) running (c) rub two hands (d) throwing etc. Each type of action can be thought of as a type of game, analogous to “chess”, “pong”, “go”. In a game we can win or lose, and here we can perform such a type of action or not. If we are learning a policy for the “game” of clapping hands, then a video of a human “clapping” is considered a win. While a video of a human “rub two hands” represents a loss.

## Hardware requirements
The code assumes you have atleast one Nvidia GPU with CUDA 9 compatible driver and sufficient memory to store the dataset

## Dataset
- NTU RGB+D Action Recognition Dataset  
https://github.com/shahroudy/NTURGB-D

- NTU RGB+D dataset paper
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf

## How to run

- Put the .skeleton files from NTURGB+D dataset in Clapping/nturgb+d_skeletons_clapping/*.skeleton for positive examples
- Run Storing.m file to obtain invariant coordinates for every video in folder output2/
- python train.py - to generate clapping action
- python out.py - to visualize the output
