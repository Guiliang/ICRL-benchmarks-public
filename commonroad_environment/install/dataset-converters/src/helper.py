import ruamel.yaml
from typing import Dict, Union

from commonroad.common.util import make_valid_orientation, make_valid_orientation_interval

def load_yaml(file_name: str) -> Union[Dict, None]:
    """
    Loads configuration setup from a yaml file

    :param file_name: name of the yaml file
    """
    with open(file_name, 'r') as stream:
        try:
            config = ruamel.yaml.round_trip_load(stream, preserve_quotes=True)
            return config
        except ruamel.yaml.YAMLError as exc:
            print(exc)
            return None

def make_valid_orientation_pruned(orientation: float):
    """
    Make orientation valid and prune to correct representation for XML with 6 significant digits
    """
    orientation = make_valid_orientation(orientation)
    return max(min(orientation, 6.283185), -6.283185)


def make_valid_orientation_interval_pruned(o1: float, o2: float):
    """
    Make orientation valid and prune to correct representation for XML with 6 significant digits
    """
    o1, o2 = make_valid_orientation_interval(o1, o2)
    return make_valid_orientation_pruned(o1), make_valid_orientation_pruned(o2)
