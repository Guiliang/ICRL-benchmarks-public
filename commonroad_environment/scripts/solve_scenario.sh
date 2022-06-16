#!/usr/bin/env bash
# Solve a scenario using a trained RL model

PICKLE_PATH=${PWD}/pickles
TEST_PATH=${PICKLE_PATH}/problem_test

NUM_CPUS=1

python -m commonroad_rl.tools.pickle_scenario.copy_files -i ${TEST_PATH} -o ${TEST_PATH} -n $NUM_CPUS -f *.pickle

# Inference
mpirun -np ${NUM_CPUS} python -m commonroad_rl.solve_stable_baselines --algo ppo2 -model ${PWD}/log/ppo2/commonroad-v0_1

## Delete subfolders
for i in $(seq 1 $NUM_CPUS)
do
        rm -rf ${TEST_PATH:?}/$((i-1))
done

