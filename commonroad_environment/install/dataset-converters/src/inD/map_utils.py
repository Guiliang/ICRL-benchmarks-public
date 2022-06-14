
__desc__ = """
Extracts planning problems from a big recording by removing a dynamic vehicle and replacing its first and last state
with ego vehicle start and goal position
"""

import os
import logging
from typing import Dict

from commonroad.scenario.scenario import Scenario, ScenarioID
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.common.file_reader import CommonRoadFileReader

LOGGER = logging.getLogger(__name__)

# has to be loaded before usage
# by calling "load_lanelet_networks"
locationId_to_lanelet_network = {}


def load_lanelet_networks(map_dir: str, ind_config: Dict) -> Dict[int, LaneletNetwork]:
    """
    Load all lanelet networks from the given path into the static variable of the file
    :param map_dir: Path to lanelet network
    :return:
    """
    # load all lanelet networks in cache
    for i, location_name in ind_config.get("locations").items():
        LOGGER.info(f"Loading lanelet network {location_name} from {map_dir}")
        map_file = os.path.join(map_dir, f"{location_name}.xml")
        locationId_to_lanelet_network[i] = CommonRoadFileReader(map_file).open_lanelet_network()
    # also return the *global* dictionary in case s.o. wants to further manipulate it
    return locationId_to_lanelet_network


def meta_scenario_from_recording(ind_config: Dict, location_id: int, recording_id: int, frame_rate=30) -> Scenario:
    """
    Generate a meta scenario from the recording meta information
    :param location_id: ID of the location in inD dataset
    :param recording_id: ID of the recording in inD dataset
    :param frame_rate: of the recording
    :return: Meta scenario, containing lanelet_network only
    """
    # compute time step from frame rate
    scenario_dt = 1 / frame_rate

    # id should not be 0 indexed, increase by one to prevent recording id = 0
    benchmark_id = f"DEU_{ind_config.get('location_benchmark_id')[location_id]}-{location_id}_{recording_id + 1}_T-1"
    scenario = Scenario(
        dt=scenario_dt,
        scenario_id=ScenarioID.from_benchmark_id(benchmark_id, scenario_version="2020a")
    )

    lanelet_network = locationId_to_lanelet_network[location_id]
    scenario.add_objects(lanelet_network)

    return scenario