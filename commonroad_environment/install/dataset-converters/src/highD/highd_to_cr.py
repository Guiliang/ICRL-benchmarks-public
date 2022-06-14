import os
import glob
import copy
import math
import numpy as np
import pandas as pd
import multiprocessing
from typing import Dict

from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.scenario.scenario import Scenario, Tag, ScenarioID

from src.highD.map_utils import get_meta_scenario, get_speed_limit, get_lane_markings, get_dt, Direction
from src.highD.obstacle_utils import generate_dynamic_obstacle
from src.planning_problem_utils import generate_planning_problem, NoCarException
from src.helper import load_yaml


def generate_scenarios_for_record(recording_meta_fn: str, tracks_meta_fn: str, tracks_fn: str,
                                  num_time_steps_scenario: int, num_planning_problems: int, keep_ego: bool,
                                  output_dir: str, highd_config: Dict, obstacle_start_at_zero: bool, downsample: int,
                                  num_vertices: int):
    """
    Generate CommonRoad scenarios with given paths to highD for a high-D recording

    :param recording_meta_fn: path to *_recordingMeta.csv
    :param tracks_meta_fn: path to *_tracksMeta.csv
    :param tracks_fn: path to *_tracks.csv
    :param num_time_steps_scenario: maximal number of time steps per CommonRoad scenario
    :param num_planning_problems: number of planning problems per CommonRoad scenario
    :param keep_ego: boolean indicating if vehicles selected for planning problem should be kept in scenario
    :param output_dir: path to store generated CommonRoad scenario files
    :param highd_config: dictionary with configuration parameters for highD scenario generation
    :param obstacle_start_at_zero: boolean indicating if the initial state of an obstacle has to have time step zero
    :param downsample: resample states of trajectories of dynamic obstacles every downsample steps
    :param num_vertices: number of waypoints of lanes
    """
    # read data frames from the three files
    recording_meta_df = pd.read_csv(recording_meta_fn, header=0)
    tracks_meta_df = pd.read_csv(tracks_meta_fn, header=0)
    tracks_df = pd.read_csv(tracks_fn, header=0)

    # generate meta scenario with lanelet network
    dt = get_dt(recording_meta_df) * downsample
    speed_limit = get_speed_limit(recording_meta_df)
    upper_lane_markings, lower_lane_markings = get_lane_markings(recording_meta_df)
    meta_scenario_upper = get_meta_scenario(dt, "DEU_MetaScenarioUpper-0_0_T-1", upper_lane_markings, speed_limit,
                                            highd_config.get("road_length"), Direction.UPPER,
                                            highd_config.get("road_offset"), num_vertices=num_vertices)
    meta_scenario_lower = get_meta_scenario(dt, "DEU_MetaScenarioLower-0_0_T-1", lower_lane_markings, speed_limit,
                                            highd_config.get("road_length"), Direction.LOWER,
                                            highd_config.get("road_offset"), num_vertices=num_vertices)

    # separate record and generate scenario for each separated part for each direction
    # (upper interstate direction / lower interstate direction)
    num_time_steps_scenario_original_dt = num_time_steps_scenario * downsample
    num_scenarios = math.ceil(max(tracks_meta_df.finalFrame) / num_time_steps_scenario_original_dt)
    for idx_1 in range(num_scenarios):
        # benchmark id format: COUNTRY_SCENE_CONFIG_PRED
        frame_start = idx_1 * num_time_steps_scenario_original_dt + (idx_1 + 1)
        frame_end = frame_start + num_time_steps_scenario_original_dt
        benchmark_id = "DEU_{0}-{1}_{2}_T-1".format(
            highd_config.get("locations")[recording_meta_df.locationId.values[0]] + "Upper",
            int(recording_meta_df.id), idx_1 + 1)
        if num_planning_problems > 1:
            benchmark_id = "C-" + benchmark_id
        try:
            generate_single_scenario(highd_config, num_planning_problems, keep_ego, output_dir, tracks_df,
                                     tracks_meta_df, meta_scenario_upper, benchmark_id, Direction.UPPER, frame_start,
                                     frame_end, obstacle_start_at_zero, downsample)
        except NoCarException as e:
            print(f"No car in this scenario: {repr(e)}. Skipping this scenario.")

        benchmark_id = "DEU_{0}-{1}_{2}_T-1".format(
            highd_config.get("locations")[recording_meta_df.locationId.values[0]] + "Lower",
            int(recording_meta_df.id), idx_1 + 1)
        if num_planning_problems > 1:
            benchmark_id = "C-" + benchmark_id
        try:
            generate_single_scenario(highd_config, num_planning_problems, keep_ego,
                                     output_dir, tracks_df, tracks_meta_df, meta_scenario_lower, benchmark_id,
                                     Direction.LOWER, frame_start, frame_end, obstacle_start_at_zero, downsample)
        except NoCarException as e:
            print(f"No car in this scenario: {repr(e)}. Skipping this scenario.")


def generate_single_scenario(highd_config: Dict, num_planning_problems: int, keep_ego: bool, output_dir: str,
                             tracks_df: pd.DataFrame, tracks_meta_df: pd.DataFrame, meta_scenario: Scenario,
                             benchmark_id: str, direction: Direction, frame_start: int, frame_end: int,
                             obstacle_start_at_zero: bool, downsample: int):
    """
    Generate a single CommonRoad scenario based on hihg-D record snippet

    :param highd_config: dictionary with configuration parameters for highD scenario generation
    :param num_planning_problems: number of planning problems per CommonRoad scenario
    :param output_dir: path to store generated CommonRoad scenario files
    :param keep_ego: boolean indicating if vehicles selected for planning problem should be kept in scenario
    :param tracks_df: single track
    :param tracks_meta_df: single meta information track
    :param meta_scenario: CommonRoad scenario with lanelet network
    :param benchmark_id: CommonRoad benchmark ID for scenario
    :param direction: indicator for upper or lower road of interstate
    :param frame_start: start of frame in time steps of record
    :param frame_end: end of frame in time steps of record
    :param obstacle_start_at_zero: boolean indicating if the initial state of an obstacle has to start
    at time step zero
    :param downsample: resample states of trajectories of dynamic obstacles every downsample steps
    :return: None
    """

    def enough_time_steps(veh_id):
        vehicle_meta = tracks_meta_df[tracks_meta_df.id == veh_id]
        if not obstacle_start_at_zero and frame_end - int(vehicle_meta.initialFrame) < 2 * downsample \
                or int(vehicle_meta.finalFrame) - frame_start < 2 * downsample:
            return False
        elif obstacle_start_at_zero and int(vehicle_meta.initialFrame) > frame_start \
            or int(vehicle_meta.finalFrame) - frame_start < 2 * downsample:
            return False
        return True

    # copy meta_scenario with lanelet networks
    scenario = copy.deepcopy(meta_scenario)
    scenario.scenario_id = ScenarioID.from_benchmark_id(benchmark_id, '2020a')

    # read tracks appear between [frame_start, frame_end]
    scenario_tracks_df = tracks_df[(tracks_df.frame >= frame_start) & (tracks_df.frame <= frame_end)]

    # generate CR obstacles
    for vehicle_id in [vehicle_id for vehicle_id in scenario_tracks_df.id.unique()
                       if vehicle_id in tracks_meta_df[tracks_meta_df.drivingDirection == direction.value].id.unique()]:
        # if appearing time steps < min_time_steps, skip vehicle
        if not enough_time_steps(vehicle_id):
            continue
        print("Generating scenario {}, vehicle id {}".format(benchmark_id, vehicle_id), end="\r")
        do = generate_dynamic_obstacle(scenario, vehicle_id, tracks_meta_df, scenario_tracks_df, frame_start, downsample)
        scenario.add_objects(do)

    # return if scenario contains no dynamic obstacle
    if len(scenario.dynamic_obstacles) == 0 or len(scenario.dynamic_obstacles) == 1 and not keep_ego:
        return

    # generate planning problems
    planning_problem_set = PlanningProblemSet()
    for idx_2 in range(num_planning_problems):
        planning_problem = generate_planning_problem(scenario, keep_ego=keep_ego)
        planning_problem_set.add_planning_problem(planning_problem)

    # rotate scenario if it is upper scenario
    if direction == Direction.UPPER:
        translation = np.array([0.0, 0.0])
        angle = np.pi
        scenario.translate_rotate(translation, angle)
        planning_problem_set.translate_rotate(translation, angle)

    # write new scenario
    tags = {Tag(tag) for tag in highd_config.get("tags")}
    fw = CommonRoadFileWriter(scenario, planning_problem_set, highd_config.get("author"),
                              highd_config.get("affiliation"), highd_config.get("source"), tags)
    filename = os.path.join(output_dir, "{}.xml".format(scenario.scenario_id))
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS, check_validity=obstacle_start_at_zero)
    print("Scenario file stored in {}".format(filename))


def create_highd_scenarios(input_dir: str, output_dir: str, num_time_steps_scenario: int, num_planning_problems: int,
                           keep_ego: bool, obstacle_start_at_zero: bool, num_processes: int = 1, downsample: int = 1,
                           num_vertices: int = 10):
    """
    Iterates over all dataset files and generates CommonRoad scenarios

    :param input_dir: path to dataset files
    :param output_dir: path to store generated CommonRoad scenario files
    :param num_time_steps_scenario: number of time steps per CommonRoad scenario
    :param num_planning_problems: number of planning problems per CommonRoad scenario
    :param keep_ego: boolean indicating if vehicles selected for planning problem should be kept in scenario
    :param obstacle_start_at_zero: boolean indicating if the initial state of an obstacle has to have time step zero
    :param num_processes: number of parallel processes to convert raw data (Optimal=60)
    :param downsample: resample states of trajectories of dynamic obstacles every downsample steps
    :param num_vertices: number of waypoints of lanes
    """
    # generate path to highd data files
    path_tracks = os.path.join(input_dir, "data/*_tracks.csv")
    path_metas = os.path.join(input_dir, "data/*_tracksMeta.csv")
    path_recording = os.path.join(input_dir, "data/*_recordingMeta.csv")

    # get all file names
    listing_tracks = sorted(glob.glob(path_tracks))
    listing_metas = sorted(glob.glob(path_metas))
    listing_recording = sorted(glob.glob(path_recording))

    highd_config = load_yaml(os.path.dirname(os.path.abspath(__file__)) + "/config.yaml")

    if num_processes < 2:
        for index, (recording_meta_fn, tracks_meta_fn, tracks_fn) in \
                enumerate(zip(listing_recording, listing_metas, listing_tracks)):
            print("=" * 80)
            print("Processing file {}...".format(tracks_fn), end='\n')
            generate_scenarios_for_record(recording_meta_fn, tracks_meta_fn, tracks_fn, num_time_steps_scenario,
                                          num_planning_problems, keep_ego, output_dir, highd_config,
                                          obstacle_start_at_zero, downsample, num_vertices)
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                generate_scenarios_for_record,
                [
                    (
                        recording_meta_fn,
                        tracks_meta_fn,
                        tracks_fn,
                        num_time_steps_scenario,
                        num_planning_problems,
                        keep_ego,
                        output_dir,
                        highd_config,
                        obstacle_start_at_zero,
                        downsample,
                        num_vertices
                    )
                    for recording_meta_fn, tracks_meta_fn, tracks_fn in \
                    zip(listing_recording, listing_metas, listing_tracks)
                ]
            )

