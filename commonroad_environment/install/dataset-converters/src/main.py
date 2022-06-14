import os
import time
import argparse
import warnings

from src.highD.highd_to_cr import create_highd_scenarios
from src.inD.ind_to_cr import create_ind_scenarios
from src.INTERACTION.interaction_to_cr import create_interaction_scenarios


def get_args() -> argparse.Namespace:
    """
    Specifies and reads command line arguments

    :return: command line arguments
    """
    parser = argparse.ArgumentParser(description="Generates CommonRoad scenarios different datasets",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str, choices=["inD", "highD", "INTERACTION"], help='Specification of dataset')
    parser.add_argument('input_dir', type=str, help='Path to dataset files')
    parser.add_argument('output_dir', type=str, help='Directory to store generated CommonRoad files')
    parser.add_argument('--num_time_steps_scenario', type=int, default=150,
                        help='Maximum number of time steps the CommonRoad scenario can be long, default=150')
    parser.add_argument('--num_planning_problems', type=int, default=1,
                        help='Number of planning problems per CommonRoad scenario, default=1')
    parser.add_argument('--keep_ego', default=False, action='store_true',
                        help='Indicator if vehicles used for planning problem should be kept in scenario, '
                             'default=False')
    parser.add_argument('--obstacle_start_at_zero', default=False, action='store_true',
                        help='Indicator if the initial state of an obstacle has to start at time step zero, '
                             'default=False')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='Number of multiple processes to convert dataset, '
                             'default=1')
    parser.add_argument('--inD_all', default=False, action='store_true',
                        help='(Only inD) Convert one CommonRoad scenario for each valid vehicle from inD dataset,'
                             ' since it has less recordings available, note that if enabled, num_time_steps_scenario'
                             ' becomes the minimal number of time steps of one CommonRoad scenario')
    parser.add_argument('--downsample', type=int, default=1, help='Decrease dt by n*dt, works only for highD converter')
    parser.add_argument('--num_vertices', type=int, default=10,
                        help='Number of lane waypoints, works only for highD converter')

    return parser


def main(args):
    start_time = time.time()

    # make output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # check parameters for specific converters
    if args.dataset != "higD" and (args.downsample != 1 or args.num_vertices != 10):
        warnings.warn("Downsample and num_vertices are only available for highD converter! Ignored")
    if args.dataset != "inD" and args.inD_all:
        warnings.warn("inD_all are only available for inD converter! Ignored")

    if args.dataset == "highD":
        create_highd_scenarios(args.input_dir, args.output_dir, args.num_time_steps_scenario,
                               args.num_planning_problems, args.keep_ego, args.obstacle_start_at_zero,
                               args.num_processes, args.downsample, args.num_vertices)
    elif args.dataset == "inD":
        if args.downsample != 1:
            warnings.warn('Downsampling only implemented for highD. Using original temporal resolution!')
        create_ind_scenarios(args.input_dir, args.output_dir, args.num_time_steps_scenario,
                             args.num_planning_problems, args.keep_ego, args.obstacle_start_at_zero,
                             num_processes=args.num_processes, inD_all=args.inD_all)
    elif args.dataset == "INTERACTION":
        if args.downsample != 1:
            warnings.warn('Downsampling only implemented for highD. Using original temporal resolution!')
        create_interaction_scenarios(args.input_dir, args.output_dir,
                                     obstacle_start_at_zero=args.obstacle_start_at_zero,
                                     num_planning_problems=args.num_planning_problems, keep_ego=args.keep_ego,
                                     num_time_steps_scenario=args.num_time_steps_scenario,
                                     num_processes=args.num_processes)
    else:
        print("Unknown dataset in command line parameter!")

    print("Elapsed time: {} s".format(time.time() - start_time), end="\r")


if __name__ == "__main__":
    # get arguments
    args = get_args().parse_args()
    main(args)
