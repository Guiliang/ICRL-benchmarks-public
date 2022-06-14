__desc__ = """
Generates dynamic obstacles for the INTERACTION conversion
"""

import numpy as np
import pandas as pd

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.prediction.prediction import TrajectoryPrediction


def get_velocity(track_df: pd.DataFrame) -> np.array:
    """
    Calculates velocity given x-velocity and y-velocity

    :param track_df: track data frame of a vehicle
    :return: array of velocities for vehicle
    """
    return np.sqrt(track_df.vx ** 2 + track_df.vy ** 2)


def get_type_obstacle_commonroad(type_agent):
    dict_conversion = {'car': ObstacleType.CAR,
                       'truck': ObstacleType.TRUCK,
                       'bus': ObstacleType.BUS,
                       'bicycle': ObstacleType.BICYCLE,
                       'motorcycle': ObstacleType.MOTORCYCLE}

    type_obstacle_CR = dict_conversion.get(type_agent, ObstacleType.UNKNOWN)

    assert type_obstacle_CR != ObstacleType.UNKNOWN, f"Found obstacle of type Unknown {type_agent}."

    return type_obstacle_CR


def generate_dynamic_obstacle(scenario: Scenario, track_df: pd.DataFrame, time_start_track: int) -> DynamicObstacle:
    length = track_df.length.values[0]
    width = track_df.width.values[0]

    dynamic_obstacle_id = scenario.generate_object_id()
    dynamic_obstacle_type = get_type_obstacle_commonroad(track_df.agent_type.values[0])
    dynamic_obstacle_shape = Rectangle(width=width, length=length)

    xs = np.array(track_df.x)
    ys = np.array(track_df.y)
    velocities = get_velocity(track_df)
    orientations = np.array(track_df.psi_rad)

    state_list = []
    for i, (x, y, v, theta) in enumerate(zip(xs, ys, velocities, orientations)):
        state_list.append(State(position=np.array([x, y]), velocity=v, orientation=theta,
                                time_step=time_start_track + i))

    dynamic_obstacle_initial_state = state_list[0]

    dynamic_obstacle_trajectory = Trajectory(time_start_track + 1, state_list[1:])
    dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)

    return DynamicObstacle(dynamic_obstacle_id, dynamic_obstacle_type, dynamic_obstacle_shape,
                           dynamic_obstacle_initial_state, dynamic_obstacle_prediction)


def generate_all_obstacles(scenario: Scenario, track_df: pd.DataFrame, obstacle_start_at_zero: bool,
                           time_start_scenario: int, time_end_scenario: int):
    # generate obstacles
    vehicle_ids = track_df.track_id.unique()

    for id_vehicle in vehicle_ids:
        """
        discard vehicles that (1) start after the scenario ends, or (2) end before the scenario starts.
        for one-shot planning scenarios, we don't consider vehicles that (3) start after time step 0 as well.
        """
        track = track_df[(track_df.track_id == id_vehicle) & (track_df.timestamp_ms >= time_start_scenario)]
        time_start_track = track.timestamp_ms.min()
        time_end_track = track.timestamp_ms.max()

        if len(track) == 0:
            continue

        def enough_time_steps():
            if not obstacle_start_at_zero and time_end_scenario - time_start_track < 2 \
                    or time_end_track - time_start_scenario < 2:
                return False
            elif obstacle_start_at_zero and time_start_track > time_start_scenario \
                    or time_end_scenario - time_start_scenario < 2:
                return False
            return True

        if not enough_time_steps():
            continue

        time_start_track -= time_start_scenario
        dynamic_obstacle = generate_dynamic_obstacle(scenario, track, int(time_start_track))

        scenario.add_objects(dynamic_obstacle)

    return scenario
