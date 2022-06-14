import warnings
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile

__author__ = "Xiao Wang"
__copyright__ = "TUM Cyber-Physical Systems Group"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Release"

__desc__ = """
Converts INTERACTION files to Commonroad Format, creating smaller Planning Problems if required
"""

import os
import glob
import copy
import multiprocessing
import numpy as np
import pandas as pd

from typing import Union, List

from commonroad.scenario.scenario import Tag, Scenario, ScenarioID
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet

from src.helper import load_yaml
from src.INTERACTION.obstacle_utils import generate_all_obstacles
from src.planning_problem_utils import generate_planning_problem, NoCarException


def generate_single_scenario(output_dir, id_segment, tags, interaction_config, dt: float, scenario_time_steps: int,
                             track_df, lanelet_network: LaneletNetwork, benchmark_id: str,
                             obstacle_start_at_zero: bool = True, keep_ego: bool = False,
                             num_planning_problems: int = 1):
    # if (id_segment + 1) % 10 == 0 or (id_segment + 1) == num_segments: print(
    #     f"\t{id_segment + 1} / {num_segments} segments processed.")

    # generate scenario of current segment
    # time of scenario
    time_start_scenario = id_segment * scenario_time_steps + 1
    time_end_scenario = (id_segment + 1) * scenario_time_steps + 1

    # create CommonRoad scenario object
    scenario = Scenario(dt=dt, scenario_id=ScenarioID.from_benchmark_id(benchmark_id, "2020a"))

    # add lanelet network to scenario
    scenario.add_objects(lanelet_network)

    # add all obstacles to scenario
    scenario = generate_all_obstacles(
        scenario, track_df, obstacle_start_at_zero, time_start_scenario, time_end_scenario
    )

    # skip if there is only a few obstacles in the scenario
    if len(scenario.dynamic_obstacles) < num_planning_problems:
        return

    # generate planning problems
    planning_problem_set = PlanningProblemSet()
    for _ in range(num_planning_problems):
        planning_problem = generate_planning_problem(scenario, keep_ego=keep_ego)
        planning_problem_set.add_planning_problem(planning_problem)

    # write new scenario
    fw = CommonRoadFileWriter(scenario, planning_problem_set, interaction_config.get("author"),
                              interaction_config.get("affiliation"), interaction_config.get("source"), tags)
    filename = os.path.join(output_dir, "{}.xml".format(scenario.scenario_id))
    if obstacle_start_at_zero is True:
        check_validity = True
    else:
        check_validity = False
    fw.write_to_file(filename, OverwriteExistingFile.ALWAYS, check_validity=check_validity)
    # print("Scenario file stored in {}".format(filename))


def generate_scenarios_for_map(location: str, map_dir: str, input_dir: str, output_dir: str, interaction_config,
                               scenario_time_steps=100, obstacle_start_at_zero: bool = True,
                               num_planning_problems: int = 1, keep_ego: bool = False):
    """
    Generate CommonRoad scenarios with given paths to INTERACTION for a folder of tracks for one map,
    each map has several tracks, each track can be separated into multiple scenarios

    :param location: location name
    :param map_dir: path the directory of pre-generated .xml map files
    :param input_dir: path to raw dataset directory
    :param output_dir: path to output directory
    :param interaction_config: configuration dictionary
    :param scenario_time_steps: maximal number of time steps per CommonRoad scenario
    :param obstacle_start_at_zero: boolean indicating if the initial state of an obstacle has to have time step zero
    :param num_planning_problems: number of planning problems per CommonRoad scenario
    :param keep_ego: boolean indicating if vehicles selected for planning problem should be kept in scenario
    :param dt:
    :return:
    """

    prefix_name = location + '_',
    path_map = f"{os.path.join(os.getcwd(), map_dir, interaction_config['maps'][location])}.xml"
    directory_data = os.path.join(input_dir, interaction_config['directory_data'][location])
    directory_output = os.path.join(os.getcwd(), output_dir, f"{location}/")

    if not os.path.exists(directory_data):
        warnings.warn(f"Directory {directory_data} does not exist, skipping this map.")
        return 0
    x_offset_tracks = interaction_config['offsets'][location]['x_offset_tracks']
    y_offset_tracks = interaction_config['offsets'][location]['y_offset_tracks']
    tags = [Tag(tag) for tag in interaction_config['tags'][location].split(' ')]
    dt = interaction_config['dt']

    # check validity of map file
    assert os.path.isfile(path_map), f"Scenarios with prefix <{prefix_name}> not created. Map file not found."

    # open map and read in scenario and planning problems (empty at the moment)
    scenario_source, _ = CommonRoadFileReader(path_map).open()

    # create output directory
    os.makedirs(directory_output, exist_ok=True)

    # get list of directories in the data directory
    path_files = sorted(glob.glob(os.path.join(directory_data, "*.csv")))
    assert len(path_files), f"Scenarios with prefix <{prefix_name}> not created. Recorded track files not found."

    # this specifies the configuration id of scenario
    id_config_scenario = 1

    # prepare lanelet network for scenarios from the given source
    lanelet_network = copy.deepcopy(scenario_source.lanelet_network)

    # iterate through record files
    for path_file in path_files:
        track_df = pd.read_csv(path_file, header=0)
        track_df["timestamp_ms"] = (track_df["timestamp_ms"] / 1000. // dt).astype(int)
        time_min = track_df.timestamp_ms.min()
        time_max = track_df.timestamp_ms.max()
        num_segments = int((time_max - time_min) / scenario_time_steps)

        # translate all positions
        track_df["x"] -= x_offset_tracks
        track_df["y"] -= y_offset_tracks

        for id_segment in range(num_segments):
            benchmark_id = "{0}_{1}_T-1".format(location, id_config_scenario)
            try:
                generate_single_scenario(
                    directory_output, id_segment, tags, interaction_config, dt,
                    scenario_time_steps, track_df, lanelet_network, benchmark_id,
                    obstacle_start_at_zero=obstacle_start_at_zero, keep_ego=keep_ego,
                    num_planning_problems=num_planning_problems
                )
                id_config_scenario += 1
            except NoCarException as e:
                print(f"No car in this scenario: {repr(e)}. Skipping this scenario.")

    id_config_scenario -= 1

    return id_config_scenario


def create_interaction_scenarios(input_dir: str, output_dir: str = "scenarios_converted/",
                                 map_dir: Union[str, None] = None, obstacle_start_at_zero: bool = True,
                                 num_planning_problems: int = 1, keep_ego: bool = False,
                                 num_time_steps_scenario: int = 150, num_processes: int = 1):
    """
    Iterates over all dataset files and generates CommonRoad scenarios

    :param input_dir: path to dataset files
    :param output_dir: path to store generated CommonRoad scenario files
    :param map_dir: path to folder with the preprocessed .xml files of the maps
    :param obstacle_start_at_zero: boolean indicating if the initial state of an obstacle has to have time step zero
    :param num_planning_problems: number of planning problems per CommonRoad scenario
    :param keep_ego: boolean indicating if vehicles selected for planning problem should be kept in scenario
    :param num_time_steps_scenario: number of time steps per CommonRoad scenario
    :param num_processes: number of parallel processes to convert raw data (Optimal=12)
    """
    if map_dir is None:
        map_dir = os.path.dirname(os.path.abspath(__file__)) + "/repaired_maps"

    # get config info
    assert os.path.exists(input_dir), f"{input_dir} folder not found!"
    assert os.path.exists(map_dir), f"{map_dir} folder not found!"

    interaction_config = load_yaml(os.path.dirname(os.path.abspath(__file__)) + "/config.yaml")
    print(f"Number of maps to be processed: {len(interaction_config['locations'])}")

    # iterate through the config and process the scenarios
    sum_scenarios = 0
    if num_processes < 2:
        for idx, location in enumerate(interaction_config['locations'].values()):
            print(f"\nProcessing {idx + 1} / {len(interaction_config['locations'])}:")

            num_scenarios = generate_scenarios_for_map(
                location,
                map_dir,
                input_dir,
                output_dir,
                interaction_config,
                scenario_time_steps=num_time_steps_scenario,
                obstacle_start_at_zero=obstacle_start_at_zero,
                num_planning_problems=num_planning_problems,
                keep_ego=keep_ego
            )
        sum_scenarios += num_scenarios

        print(f"""\nGenerated scenarios: {sum_scenarios}""")
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(
                generate_scenarios_for_map,
                [
                    (
                        location,
                        map_dir,
                        input_dir,
                        output_dir,
                        interaction_config,
                        num_time_steps_scenario,
                        obstacle_start_at_zero,
                        num_planning_problems,
                        keep_ego
                    ) for idx, location in enumerate(interaction_config['locations'].values())
                ]
            )
