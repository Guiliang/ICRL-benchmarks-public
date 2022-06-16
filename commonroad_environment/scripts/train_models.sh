#!/bin/bash
# Run like this to train in parallel:
#  nohup bash train_models 1 hybrid > train_models_1_hybrid.out &
# To start multiple trainings in parallel, replace the 1 or reward prefix by other values.
# Note that the first value is the seed as well as the number of the created model
# If there are n models in log/<prefix>_reward already, run the script with n+1 (including 1 if 0 models so far)
#
# Usage: scripts/train_models.sh <seed> <reward_type>
# Example: scripts/train_models.sh 1 hybrid
#
# Settings:
MAKE_GIF="false"
GIF_SCENARIO="DEU_AAH-4_0_T-1/DEU_AAH-4_1000_T-1_ts_"
AUTO_CONFIGURE_GOAL_OBSERVATIONS="false"
OUT_PATH="."
STEPS=1000000
NUM_CPUS=16
PICKLE_PATH="pickles"

while getopts ":co:s:p:i:" opt; do
  case $opt in
    s)
      STEPS=$OPTARG
      ;;
    c)
      AUTO_CONFIGURE_GOAL_OBSERVATIONS="true"
      ;;
    i)
      PICKLE_PATH=$OPTARG
      ;;
    o)
      OUT_PATH=$OPTARG
      ;;
    p)
      NUM_CPUS=$OPTARG
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

# Check arguments
if [ "$1" = "" ] || [ "$2" = "" ]; then
  echo "Usage: scripts/train_models.sh <seed> <reward_type>"
  exit 1
fi

SEED=$1
REWARD="${2}_reward"

if [ "$AUTO_CONFIGURE_GOAL_OBSERVATIONS" = true ]; then
  #Configure goal_observations
  python commonroad_rl/tools/configure_goal_observations.py -c
fi

#Training and Evaluation
echo "Training model:"
scripts/run_ppo2.sh -s "$STEPS" -o "$OUT_PATH" -p "$NUM_CPUS" -i "$PICKLE_PATH" "$SEED" "$REWARD"
echo "Evaluating model:"
scripts/play_model.sh "$SEED" "$REWARD"
echo "Plotting learning-curve:"
python commonroad_rl/plot_learning_curves.py -f "$OUT_PATH/log/$REWARD/ppo2" -model "commonroad-v1_$SEED" -legend "$REWARD - $SEED" --smooth

#to be used for combining generated images into a singleton GIF
if [ "$MAKE_GIF" = true ]; then
  echo "Making GIF:"
  mkdir -p "$OUT_PATH/res/$REWARD/$SEED"
  ffmpeg -i "$OUT_PATH/img/$REWARD/ppo2/commonroad-v1_$SEED/${GIF_SCENARIO}%03d.png" -vf palettegen "$OUT_PATH/res/$REWARD/$SEED/palette.png"
  ffmpeg -i "$OUT_PATH/img/$REWARD/ppo2/commonroad-v1_$SEED/${GIF_SCENARIO}%03d.png" -i "$OUT_PATH/res/$REWARD/$SEED/palette.png" -lavfi "paletteuse" -y "$OUT_PATH/res/output_${REWARD}_${SEED}.gif"
fi
