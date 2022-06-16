# Replay a trained RL model
# example invocation
# $ scripts/play_model.sh 1 default_reward
# where the script plays the model stored in log/default_reward/ppo2/commonroad_v1
if [ "$1" = "" ] || [ "$2" = "" ]; then
  echo "Usage: scripts/play_model.sh <model_num> <reward_type>"
  exit 1
fi

MODEL_NUM=$1
REWARD=$2
NUM_CPUS=2

PICKLE_PATH=${PWD}/pickles
TEST_PATH=${PICKLE_PATH}/problem_test
MODEL_PATH=${PWD}/log/$REWARD/ppo2/commonroad-v1_$MODEL_NUM
#VIZ_PATH=${PWD}/img/$REWARD/ppo2/$MODEL_NUM

echo "$TEST_PATH"

python -m commonroad_rl.tools.pickle_scenario.copy_files -i "${TEST_PATH}" -o "${TEST_PATH}" -n $NUM_CPUS -f "*.pickle"

# Play
mpirun -np ${NUM_CPUS} python -m commonroad_rl.evaluate_model --algo ppo2 \
 -model "$MODEL_PATH" \
 -i "$PICKLE_PATH" -mpi -1 -st 15 \
 -nr

## Delete subfolders
for i in $(seq 1 $NUM_CPUS)
do
    rm -rf "${TEST_PATH:?}"/$((i-1))
done

