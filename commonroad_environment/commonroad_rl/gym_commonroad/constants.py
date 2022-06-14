"""
Module for CommonRoad Gym environment related constants
"""
# Lanelet parameters
from commonroad_rl.gym_commonroad.utils.scenario_io import get_project_root

# Visualization parameters
DRAW_PARAMS = {
    "draw_shape": True,
    "draw_icon": True,
    "draw_bounding_box": True,
    "trajectory_steps": 2,
    "show_label": False,
    "occupancy": {
        "draw_occupancies": 0,
        "shape": {
            "rectangle": {
                "opacity": 0.2,
                "facecolor": "#fa0200",
                "edgecolor": "#0066cc",
                "linewidth": 0.5,
                "zorder": 18,
            }
        },
    },
    "shape": {
        "rectangle": {
            "opacity": 1.0,
            "facecolor": "#fa0200",
            "edgecolor": "#831d20",
            "linewidth": 0.5,
            "zorder": 20,
        }
    },
}

# Path
ROOT_STR = str(get_project_root())

PATH_PARAMS = {
    "visualization": ROOT_STR + "/img",
    "pickles": ROOT_STR + "/pickles",
    "meta_scenario": ROOT_STR + "/pickles/meta_scenario",
    "train_reset_config": ROOT_STR + "/pickles/problem_train",
    "test_reset_config": ROOT_STR + "/pickles/problem_test",
    "log": ROOT_STR + "/log",
    "commonroad_solution": ROOT_STR + "/cr_solution",
    "configs": {"commonroad-v1": ROOT_STR + "/commonroad_rl/gym_commonroad/configs.yaml",
                "cr_monitor-v0": ROOT_STR + "/commonroad_rl/gym_commonroad/configs.yaml"}
}
