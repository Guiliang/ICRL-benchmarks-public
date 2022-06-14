import math
import numpy as np
from typing import Union
from pandas import DataFrame, Series

from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State, Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.scenario import Scenario

obstacle_class_dict = {
    'Truck': ObstacleType.TRUCK,
    'Car': ObstacleType.CAR
}


def get_velocity(track_df: DataFrame) -> np.array:
    """
    Calculates velocity given x-velocity and y-velocity

    :param track_df: track data frame of a vehicle
    :return: array of velocities for vehicle
    """
    return np.sqrt(track_df.xVelocity ** 2 + track_df.yVelocity ** 2)


def get_orientation(track_df: DataFrame) -> np.array:
    """
    Calculates orientation given x-velocity and y-velocity

    :param track_df: track data frame of a vehicle
    :return: array of orientations for vehicle
    """
    return np.arctan2(-track_df.yVelocity, track_df.xVelocity)


def get_acceleration(track_df: DataFrame) -> np.array:
    """
    Calculates acceleration given x-acceleration and y-acceleration

    :param track_df: track data frame of a vehicle
    :return: array of accelerations for vehicle
    """
    return np.sqrt(track_df.xAcceleration ** 2 + track_df.yAcceleration ** 2)


def generate_dynamic_obstacle(scenario: Scenario, vehicle_id: int, tracks_meta_df: DataFrame,
                              tracks_df: DataFrame, time_step_correction: int, downsample: int) -> DynamicObstacle:
    """

    :param scenario: CommonRoad scenario
    :param vehicle_id: ID of obstacle to generate
    :param tracks_meta_df: track meta information data frames
    :param tracks_df: track data frames
    :return: CommonRoad dynamic obstacle
    """

    vehicle_meta = tracks_meta_df[tracks_meta_df.id == vehicle_id]
    vehicle_tracks = tracks_df[tracks_df.id == vehicle_id]

    length = vehicle_meta.width.values[0]
    width = vehicle_meta.height.values[0]

    initial_time_step_cr = math.ceil((int(vehicle_tracks.frame.values[0]) - time_step_correction) / downsample)
    initial_time_step_cr = int(initial_time_step_cr)
    initial_frame = initial_time_step_cr * downsample
    dynamic_obstacle_id = scenario.generate_object_id()
    dynamic_obstacle_type = obstacle_class_dict[vehicle_meta['class'].values[0]]
    dynamic_obstacle_shape = Rectangle(width=width, length=length)

    xs = np.array(vehicle_tracks.x)
    ys = np.array(-vehicle_tracks.y)
    velocities = get_velocity(vehicle_tracks)
    orientations = get_orientation(vehicle_tracks)
    accelerations = get_acceleration(vehicle_tracks)

    state_list = []
    for cr_timestep, frame_idx in enumerate(range(0, xs.shape[0], downsample)):
        x = xs[frame_idx]
        y = ys[frame_idx]
        v = velocities.values[frame_idx]
        theta = orientations.values[frame_idx]
        a = accelerations.values[frame_idx]
        state_list.append(State(position=np.array([x, y]), velocity=v, orientation=theta,
                                time_step=cr_timestep + initial_time_step_cr))

    dynamic_obstacle_initial_state = state_list[0]

    dynamic_obstacle_trajectory = Trajectory(initial_time_step_cr + 1, state_list[1:])
    dynamic_obstacle_prediction = TrajectoryPrediction(dynamic_obstacle_trajectory, dynamic_obstacle_shape)

    return DynamicObstacle(dynamic_obstacle_id, dynamic_obstacle_type, dynamic_obstacle_shape,
                           dynamic_obstacle_initial_state, dynamic_obstacle_prediction)
