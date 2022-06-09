import os

import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv

# =========================================================================== #
#                         Inverted Pendulum Wall                              #
# =========================================================================== #


class InvertedPendulumWall(InvertedPendulumEnv):

    def step(self, a):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        obs = self._get_obs()
        notdone = np.isfinite(obs).all() and (np.abs(obs[1]) <= 0.2)
        done = not notdone
        info = {'x_position': xposafter}
        if xposafter <= -0.01:
            reward = 1
        elif xposafter >= 0:
            reward = 0.1
        else:
            reward = (-xposafter/0.01)*0.9+0.1
        return obs, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
