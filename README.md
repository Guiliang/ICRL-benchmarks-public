# ICRL-benchmarks-public
This is the code for the paper "[Benchmarking Constraint Inference in Inverse Reinforcement Learning](https://openreview.net/forum?id=vINj_Hv9szL)" published in ICLR 2023. Note that:
1. Our benchmark rely on [Mujoco](https://mujoco.org/) and [CommonRoad](https://commonroad.in.tum.de/commonroad-rl). These environments are publicly available. 
2. The implementation of the baselines are based on the code from [iclr](https://github.com/shehryar-malik/icrl).



## Setup Python Virtual Environment
1. Make sure you have [downloaded & installed (mini)conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) before proceeding.
2. Create conda environment and install the packages:
```
mkdir ./save_model
mkdir ./evaluate_model
conda env create -n cn-py37 python=3.7 -f python_environment.yml
conda activate cn-py37
```
3. Install [Pytorch](https://pytorch.org/) in the conda env.

## Download the Expert Data
Note that we have generated the expert data for the ease of usage, but users can generate their own dataset by adding extra settings (we will show how to generate the expert data later).
```
cd ./data
wget https://cs.uwaterloo.ca/~ppoupart/datasets/expert_data.zip
unzip expert_data.zip
rm expert_data.zip
cd ../
```

## Tutorials
Now we are ready to run ICRL baselines on our benchmark. For more details, please follow the tutorial below.
### Part I. [***ICRL in Virtual Environment***](./virtual_env_tutorial.md).
### Part II. [***ICRL in Realistic Environment***](./realisitic_env_tutorial.md).
### Part III. [***ICRL in Discrete Environment***](./discrete_env_tutorial.md).

