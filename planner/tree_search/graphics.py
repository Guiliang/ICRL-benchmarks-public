import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from planner.tree_search.tree_utils import constrain, remap


class TreePlot(object):
    def __init__(self, planner, max_depth=4):
        self.planner = planner
        self.actions = planner.env.action_space.n
        self.max_depth = max_depth
        self.total_count = sum(c.count for c in self.planner.root.children.values())

    def plot(self, filename, title=None, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        self._plot_node(self.planner.root, [0, 0], ax)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        if title:
            plt.title(title)
        ax.axis('off')

        if filename is not None:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filename, dpi=300, figsize=(10, 10))

    def _plot_node(self, node, pos, ax, depth=0):
        if depth > self.max_depth:
            return
        for a in range(self.actions):
            if a in node.children:
                child = node.children[a]
                if not child.count:
                    continue
                d = 1 / self.actions ** depth
                pos_child = [pos[0] - d / 2 + a / (self.actions - 1) * d, pos[1] - 1 / self.max_depth]
                width = constrain(remap(child.count, (1, self.total_count), (0.5, 4)), 0.5, 4)
                ax.plot([pos[0], pos_child[0]], [pos[1], pos_child[1]], 'k', linewidth=width, solid_capstyle='round')
                self._plot_node(child, pos_child, ax, depth + 1)

    def plot_to_writer(self, writer, epoch=0, figsize=None, show=False):
        fig = plt.figure(figsize=figsize, tight_layout=True)
        ax = fig.add_subplot(111)

        title = "Expanded_tree"
        self.plot(filename=None, title=title, ax=ax)

        # Figure export
        fig.canvas.draw()
        data_str = fig.canvas.tostring_rgb()
        if writer:
            data = np.fromstring(data_str, dtype=np.uint8, sep='')
            data = np.rollaxis(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)), 2, 0)
            writer.add_image(title, data, epoch)
        if show:
            plt.show()
        plt.close()

