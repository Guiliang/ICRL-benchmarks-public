#!/usr/bin/env bash
# Run a training of the RL model via td3
# example invocation
# $ scripts/run_ppo2.sh 1 hybrid_reward
# where the script is run with seed 1 and the hybrid_reward

PICKLE_PATH=./pickles
TRAIN_PATH=${PICKLE_PATH}/problem_train
TEST_PATH=${PICKLE_PATH}/problem_test
LOG_PATH=./log/$2

NUM_CPUS=8

# Split pickle files into subfolders
python -m commonroad_rl.tools.pickle_scenario.copy_files -i ${TRAIN_PATH} -o ${TRAIN_PATH} -n $NUM_CPUS -f *.pickle -d


# Train
python -m commonroad_rl.run_stable_baselines --algo td3 --seed $1 --eval-freq 1000 \
 -f $LOG_PATH --n_envs $NUM_CPUS -n 1000000 --env-kwargs reward_type:"str('$2')" \
 --save-freq 10000 --info_keywords is_collision is_time_out is_off_road is_friction_violation is_goal_reached
 # -i ../log/ppo2/commonroad-v0_4/best_model.zip

# Delete subfolders
for i in $(seq 1 $NUM_CPUS)
do
    rm -rf ${TRAIN_PATH}/$((i-1))
done

