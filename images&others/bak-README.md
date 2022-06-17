# CIRL-benchmarks-public

This benchmark relies on [Mujoco](https://mujoco.org/) and [CommonRoad](https://commonroad.in.tum.de/commonroad-rl). These two environments are publicly available.


## Environment Setup

### Step 1: Setup Python Virtual Environment
Make sure you have [downloaded & installed conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) before proceeding.
```
mkdir ./save_model
mkdir ./evaluate_model
conda env create -n cn-py37 python=3.7 -f python_environment.yml
conda activate cn-py37
```

###  Step 2: Setup MuJoCo (for virtual environments)
To run the virtual environment, you need to set up MuJoCo.
1. Download the MuJoCo version 2.1 binaries for Linux or OSX. 
2. Extract the downloaded mujoco210 directory into ~/.mujoco/mujoco210.
3. Install and use [mujoco-py](https://github.com/openai/mujoco-py).
```
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -e ./mujuco_environment
```
We **highly recommend** you to ensure the MuJoCo is indeed working by running testing examples in [mujoco-py](https://github.com/openai/mujoco-py). In most case, you need to run:
```
import mujoco_py
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
```
### 3 Setup CommonRoad (for realistic environments)
To run the virtual environment, you need to set up CommonRoad by following the instructions in [commonroad-rl](https://gitlab.lrz.de/tum-cps/commonroad-rl.git):
1. Download the environment
```
git clone https://gitlab.lrz.de/tum-cps/commonroad-rl.git
mv commonroad-rl/ commonroad_environment
cd commonroad_environment/
```
2. Install build packages and submodules
```
sudo apt-get update
sudo apt-get install build-essential make cmake
git submodule init
git submodule update --recursive || exit_with_error "Update submodules failed"
```
3. Install with sudo rights (Check [commonroad-rl](https://gitlab.lrz.de/tum-cps/commonroad-rl.git) about installing **without** sudo rights)
```
bash scripts/install.sh -e cn-py37
```
**Now the environment should be ready to try!** 
4. [**Run with the Full HighD Data**] Get the full dataset and Preprocess.  
- Our repository uses some data examples from [commonroad-rl tutorial](https://gitlab.lrz.de/tum-cps/commonroad-rl/-/tree/master/commonroad_rl/tutorials/data). To build the full environments, you need to apply for the HighD dataset from [here](https://www.highd-dataset.com/). **The dataset is free for not non-commercial use**.
- After you received the data, do some preprocess according to [Tutorial 01 - Data Preprocessing](https://gitlab.lrz.de/tum-cps/commonroad-rl/-/blob/master/commonroad_rl/tutorials/Tutorial%2001%20-%20Data%20Preprocessing.ipynb). We show a brief version as follow:  

Once you have downloaded the data, extract all the .csv (e.g., `03_recordingMeta.csv`, `03_tracks.csv`, `03_tracksMeta.csv`) files to the folder`CIRL-benchmarks-public/data/highD/raw/data/`, and then
```
cd $YourProjectDir/CIRL-benchmarks-public/commonroad_environment/
mkdir ./install
cd ./install
git clone https://gitlab.lrz.de/tum-cps/dataset-converters.git
cd dataset-converters
pip install -r requirements.txt

# transfer raw data to .xml files
python -m src.main highD ../../../data/highD/raw/ ../../../data/highD/xmls/ --num_time_steps_scenario 1000

# compute the .pickle files
cd $YourProjectDir/CIRL-benchmarks-public/commonroad_environment/commonroad_rl
python -m commonroad_rl.tools.pickle_scenario.xml_to_pickle -i ../../data/highD/xmls -o ../../data/highD/pickles

# split the dataset
python -m commonroad_rl.utils_run.split_dataset -i ../../data/highD/pickles/problem -otrain ../../data/highD/pickles/problem_train -otest ../../data/highD/pickles/problem_test -tr_r 0.7

# scatter dataset for multiple processes
python -m commonroad_rl.tools.pickle_scenario.copy_files -i ../../data/highD/pickles/problem_train -o ../../data/highD/pickles/problem_train_split -f *.pickle -n 5
```
