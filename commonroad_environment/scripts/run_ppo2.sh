#!/usr/bin/env bash
# Run a training of the RL model via ppo2
# example invocation
# $ scripts/run_ppo2.sh 1 hybrid_reward
# where the script is run with seed 1 and the hybrid_reward


PICKLE_PATH="./pickles"
OUT_PATH="."

SEED=$1
REWARD=$2
NUM_CPUS=32
STEPS=1000000

while getopts ":p:o:s:i:" opt; do
  case $opt in
    i)
      PICKLE_PATH=$OPTARG
      ;;
    s)
      STEPS=$OPTARG
      ;;
    p)
      NUM_CPUS=$OPTARG
      ;;
    o)
      OUT_PATH=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND-1))

if [ "$1" = "" ] || [ "$2" = "" ]; then
  echo "Usage: scripts/run_ppo2.sh <seed> <reward_type>"
  exit 1
fi

SEED=$1
REWARD=$2

META_PATH="${PICKLE_PATH}/meta_scenario"
TRAIN_PATH="${PICKLE_PATH}/problem_train"
TEST_PATH="${PICKLE_PATH}/problem_test"
LOG_PATH="$OUT_PATH/log"

# Split pickle files into subfolders
python -m commonroad_rl.tools.pickle_scenario.copy_files -i "${TRAIN_PATH}" -o "${TRAIN_PATH}" -n "$NUM_CPUS" -f "*.pickle" -d
python -m commonroad_rl.tools.pickle_scenario.copy_files -i "${TEST_PATH}" -o "${TEST_PATH}" -n "$NUM_CPUS" -f "*.pickle" -d


# Train
python -m commonroad_rl.train_model --uuid none --algo ppo2 --seed "$SEED" --eval-freq -1 -f "$LOG_PATH" --n_envs "$NUM_CPUS" -n "$STEPS" \
  --env-kwargs reward_type:"str('$REWARD')"\
  meta_scenario_path:"str('$META_PATH')" \
  train_reset_config_path:"str('${TRAIN_PATH}')" \
  test_reset_config_path:"str('${TEST_PATH}')"  --save-freq 10000 \
  --info_keywords is_collision is_time_out is_off_road is_friction_violation is_goal_reached \
 # --optimize-reward-configs --n-trials 200 --n-jobs 1 --eval-episodes 200 --guided\ # set eval frequency to a value ~10% of trainingsteps
# --debug
# -i ../log/ppo2/commonroad-v0_4/best_model.zip

Delete subfolders
for i in $(seq 1 "$NUM_CPUS")
do
   rm -rf "${TRAIN_PATH:?}"/$((i-1))
   rm -rf "${TEST_PATH:?}"/$((i-1))
done