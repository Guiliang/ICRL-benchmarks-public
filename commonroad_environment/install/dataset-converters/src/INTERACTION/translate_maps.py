import os

import numpy as np

from argparse import ArgumentParser

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile, Tag
from src.helper import load_yaml


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("-i", default="./repaired_maps", dest="input",
                        help="Path to directory containing commonroad formatted lanelets (converts all files)")
    parser.add_argument("-o", default="./repaired_maps/translated", help="Path to output directory", dest="output")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    os.makedirs(args.output, exist_ok=True)

    interaction_config = load_yaml(os.path.dirname(os.path.abspath(__file__)) + "/config.yaml")

    author = interaction_config["author"]
    affiliation = interaction_config["affiliation"]
    source = interaction_config["source"]

    for location in interaction_config["locations"].values():
        file_path = os.path.join(args.input, f"{interaction_config['maps'][location]}.xml")
        try:
            scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
            x_offset_lanelets = interaction_config["offsets"][location]["x_offset_lanelets"]
            y_offset_lanelets = interaction_config["offsets"][location]["y_offset_lanelets"]
            tags = [Tag(tag) for tag in interaction_config['tags'][location].split(' ')]

            scenario.translate_rotate(np.array([-x_offset_lanelets, -y_offset_lanelets]), 0)

            file_writer = CommonRoadFileWriter(scenario, planning_problem_set, author, affiliation, source, tags)
            output_file = os.path.join(args.output, f"{interaction_config['maps'][location]}.xml")
            file_writer.write_to_file(output_file, OverwriteExistingFile.ALWAYS)
        except:
            continue

