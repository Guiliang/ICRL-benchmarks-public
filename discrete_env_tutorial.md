## Run Discrete Benchmark

###  Step 1: Setup MuJoCo (skip it if you have gone through the virtual environment tutorial)
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

###  Step 2 (optionally): Train expert agents.
Note that We have a total of 4 different settings.
```
# step in the dir containing the "main" files.
cd ./interface/

# run PPO-Lag knowing the ground-truth
python train_policy.py ../config/mujoco_WGW-v0/train_ppo_lag_WGW-v0-setting1.yaml -n 5 -s 123
python train_policy.py ../config/mujoco_WGW-v0/train_ppo_lag_WGW-v0-setting2.yaml -n 5 -s 123
python train_policy.py ../config/mujoco_WGW-v0/train_ppo_lag_WGW-v0-setting3.yaml -n 5 -s 123
python train_policy.py ../config/mujoco_WGW-v0/train_ppo_lag_WGW-v0-setting4.yaml -n 5 -s 123
```

### Step 3: Run the ICLR algorithms
Note that:
1. This is to reproduce the results in the Section 6.2 of our paper. 
2. We have a total of 4 different settings.
3. Random seeds are not required since the environments and models are deterministic.
```
# step in the dir containing the "main" files.
cd ./interface/

# run GACL
python train_gail.py ../config/mujoco_WGW-v0/train_GAIL_WGW-v0-setting1.yaml -n 5
python train_gail.py ../config/mujoco_WGW-v0/train_GAIL_WGW-v0-setting2.yaml -n 5
python train_gail.py ../config/mujoco_WGW-v0/train_GAIL_WGW-v0-setting3.yaml -n 5
python train_gail.py ../config/mujoco_WGW-v0/train_GAIL_WGW-v0-setting4.yaml -n 5

# run BC2L
python train_icrl.py ../config/mujoco_WGW-v0/train_Binary_WGW-v0-setting1.yaml -n 5
python train_icrl.py ../config/mujoco_WGW-v0/train_Binary_WGW-v0-setting2.yaml -n 5
python train_icrl.py ../config/mujoco_WGW-v0/train_Binary_WGW-v0-setting3.yaml -n 5 
python train_icrl.py ../config/mujoco_WGW-v0/train_Binary_WGW-v0-setting4.yaml -n 5

# run MECL
python train_icrl.py ../config/mujoco_WGW-v0/train_ICRL_WGW-v0-setting1.yaml -n 5
python train_icrl.py ../config/mujoco_WGW-v0/train_ICRL_WGW-v0-setting2.yaml -n 5
python train_icrl.py ../config/mujoco_WGW-v0/train_ICRL_WGW-v0-setting3.yaml -n 5
python train_icrl.py ../config/mujoco_WGW-v0/train_ICRL_WGW-v0-setting4.yaml -n 5

# run VICRL
python train_icrl.py ../config/mujoco_WGW-v0/train_VICRL_WGW-v0-setting1.yaml -n 5
python train_icrl.py ../config/mujoco_WGW-v0/train_VICRL_WGW-v0-setting2.yaml -n 5
python train_icrl.py ../config/mujoco_WGW-v0/train_VICRL_WGW-v0-setting3.yaml -n 5
python train_icrl.py ../config/mujoco_WGW-v0/train_VICRL_WGW-v0-setting4.yaml -n 5
```