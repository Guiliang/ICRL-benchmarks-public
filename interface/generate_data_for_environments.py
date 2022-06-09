import os
import pickle
import random

from utils.data_utils import read_args, load_config


def generate_data_for_environment(args, obs_percentage=0.5):
    config, debug_mode, log_file_path = load_config(args)

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''
    if debug_mode:
        debug_msg = 'debug-'
    read_data_path = os.path.join(config['source_path'], 'problem_test')
    save_data_path = os.path.join(config['source_path'], 'problem_test_percent_{0}'.format(obs_percentage))
    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)
    for data_path in os.listdir(read_data_path):
        with open(os.path.join(read_data_path, data_path), 'rb') as read_file:
            environment_data = pickle.load(read_file)
            all_obstacle = environment_data['obstacle']
            total_num = len(all_obstacle)
            # print(total_num)
            save_idxs = []
            save_obstacle = []
            while len(save_idxs) < int(total_num * (1 - obs_percentage)):
                random_idx = random.randint(0, total_num - 1)
                if random_idx not in save_idxs:
                    save_idxs.append(random_idx)
            save_idxs.sort(reverse=False)
            for idx in save_idxs:
                save_obstacle.append(all_obstacle[idx])
            environment_data['obstacle'] = save_obstacle
            # total_num = len(all_obstacle)
            # obs_num = total_num
            # for i in range(int(total_num*obs_percentage)):
            #     random_idx = random.randint(0, obs_num - 1)
            #     del all_obstacle[random_idx]
            #     obs_num = len(all_obstacle)

        with open(os.path.join(save_data_path, data_path), 'wb') as save_file:
            pickle.dump(environment_data, save_file)
        with open(os.path.join(save_data_path, data_path.replace('.pickle', '.txt')), 'w') as save_file:
            save_file.write(str(save_idxs))


if __name__ == "__main__":
    args = read_args()

    generate_data_for_environment(args)
