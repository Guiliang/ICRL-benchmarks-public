# this code creates gif for random scenario from INTERACTION
__author__ = "Edmond Irani Liu"
__copyright__ = "TUM Cyber-Physical Systems Group"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Release"

import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.draw_dispatch_cr import draw_object


if __name__ == "__main__":
    random.seed()

    # num of gifs to be created
    num_gifs = 12
    # steps in each gif, 1 second = 10 steps
    steps_plot = 40
    # size of plt.figure
    size_figure = (20, 20)
    # margin for plotting figures
    margin = 10

    directory_file = "/data/interaction/cr_scenarios_tum_cps/"

    directory_gif = "./gif/"

    path_scenarios = glob.glob(os.path.join(directory_file, "*/*.xml"))
    num_scenarios = len(path_scenarios)

    print(f"{num_gifs} gifs to be generated.")

    cnt_processed = 0
    for _ in range(num_gifs):
        # get random idx
        idx_rnd = random.randint(1, num_scenarios) - 1
        path_scenrio = path_scenarios[idx_rnd]
        name_scenrio = path_scenrio.split('/')[-1].split('.')[0]
        
        directory_scenario = os.path.join(directory_gif, name_scenrio + '/')
        if not os.path.exists(directory_scenario):
            os.makedirs(directory_scenario)
        
        # read the scenario
        scenario, planning_problem_set = CommonRoadFileReader(path_scenrio).open()

        list_vertice_x = []
        list_vertice_y = []

        for lanelet in scenario.lanelet_network.lanelets:
            vertice_center = lanelet.center_vertices
            list_vertice_x.extend(list(vertice_center[:, 0])) 
            list_vertice_y.extend(list(vertice_center[:, 1]))

        x_min, x_max = min(list_vertice_x), max(list_vertice_x)
        y_min, y_max = min(list_vertice_y), max(list_vertice_y)  
        
        plt.figure(figsize=size_figure)
        
        # draw the scenario for each time step and save
        for i in range(0, steps_plot):
            plt.clf()
            draw_object(scenario, plot_limits = [x_min - margin, x_max + margin, y_min - margin, y_max + margin], 
                        draw_params={'time_begin': i})
            draw_object(planning_problem_set)
            plt.gca().set_aspect('equal')

            path_save = os.path.join(directory_scenario, name_scenrio + f"_{i}.png")
            plt.savefig(path_save, format='png', bbox_inches='tight', transparent=False)
        
        path_imgs = glob.glob(os.path.join(directory_scenario, "*.png"))
        list_names_imgs = [path_img.split("/")[-1] for path_img in path_imgs]
        list_names_imgs.sort(key=lambda name: int(name.split('_')[-1].split('.')[0]))

        frames = []
        for names_imgs in list_names_imgs:
            new_frame = Image.open(os.path.join(directory_scenario, names_imgs))
            frames.append(new_frame)

        frames[0].save(os.path.join(directory_gif, list_names_imgs[0][:-6] + '.gif'), format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=100, loop=0)
        cnt_processed += 1
        print(f"{cnt_processed} / {num_gifs} processed.")