import collections
import copy
import logging

import gym
from gym.core import Env
from abc import ABC, abstractmethod

from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


class Configurable(object):
    """
        This class is a container for a configuration dictionary.
        It allows to provide a default_config function with pre-filled configuration.
        When provided with an input configuration, the default one will recursively be updated,
        and the input configuration will also be updated with the resulting configuration.
    """

    def __init__(self, config=None):
        self.config = self.default_config()
        if config:
            # Override default config with variant
            Configurable.rec_update(self.config, config)
            # Override incomplete variant with completed variant
            Configurable.rec_update(config, self.config)

    def update_config(self, config):
        Configurable.rec_update(self.config, config)

    @classmethod
    def default_config(cls):
        """
            Override this function to provide the default configuration of the child class
        :return: a configuration dictionary
        """
        return {}

    @staticmethod
    def rec_update(d, u):
        """
            Recursive update of a mapping
        :param d: a mapping
        :param u: a mapping
        :return: d updated recursively with u
        """
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = Configurable.rec_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d


class AbstractAgent(Configurable, ABC):

    def __init__(self, config=None):
        super(AbstractAgent, self).__init__(config)
        self.writer = None  # Tensorboard writer
        self.directoy = None  # Run directory

    """
        An abstract class specifying the interface of a generic agent.
    """

    @abstractmethod
    def record(self, state, action, reward, next_state, done, info):
        """
            Record a transition of the environment to update the agent
        :param state: s, the current state of the agent
        :param action: a, the action performed
        :param reward: r(s, a), the reward collected
        :param next_state: s', the new state of the agent after the action was performed
        :param done: whether the next state is terminal
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, state):
        """
            Pick an action
        :param state: s, the current state of the agent
        :return: a, the action to perform
        """
        raise NotImplementedError()

    def plan(self, previous_actions, prior_policy):
        """
        Plan an optimal trajectory from an initial state.
        :param prior_policy: the prior policy
        :param previous_actions: previously performed actions
        :return: [a0, a1, a2...], a sequence of actions to perform
        """
        return None

    @abstractmethod
    def reset(self):
        """
            Reset the agent to its initial internal state
        """
        raise NotImplementedError()

    @abstractmethod
    def seed(self, seed=None):
        """
            Seed the agent's random number generator
        :param seed: the seed to be used to generate random numbers
        :return: the used seed
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, filename):
        """
            Save the model parameters to a file
        :param str filename: the path of the file to save the model parameters in
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, filename):
        """
            Load the model parameters from a file
        :param str filename: the path of the file to load the model parameters from
        """
        raise NotImplementedError()

    def eval(self):
        """
            Set to testing mode. Disable any unnecessary exploration.
        """
        pass

    def set_writer(self, writer):
        """
            Set a tensorboard writer to log the agent internal variables.
        :param SummaryWriter writer: a summary writer
        """
        self.writer = writer

    def set_directory(self, directory):
        self.directory = directory

    def set_time(self, time):
        """ Set a local time, to control the agent internal schedules (e.g. exploration) """
        pass


class AbstractStochasticAgent(AbstractAgent):
    """
        Agents that implement a stochastic policy
    """

    def action_distribution(self, state):
        """
            Compute the distribution of actions for a given state
        :param state: the current state
        :return: a dictionary {action:probability}
        """
        raise NotImplementedError()


def safe_deepcopy_env(obj, up_k=''):
    """
        Perform a deep copy of an environment but without copying its viewer.
    """
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        print("Copying {0}".format(up_k + '/' + k))
        if k not in ['viewer', '_monitor', 'grid_render', 'video_recorder', '_record_video_wrapper',
                     'class_attributes']:
            if isinstance(v, gym.Env) or isinstance(v, VecEnv) or isinstance(v, PyMjModel) or isinstance(v, PyMjModel):
                setattr(result, k, safe_deepcopy_env(v, up_k + '/' + k))
            elif k == 'envs':
                setattr(result, k, [])
                for i in range(len(v)):
                    if isinstance(v[i], gym.Wrapper):
                        result[k][i] = safe_deepcopy_env(v[i], up_k + '/' + k)
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result


def preprocess_env(env, preprocessor_configs):
    """
        Apply a series of pre-processes to an environment, before it is used by an agent.
    :param env: an environment
    :param preprocessor_configs: a list of preprocessor configs
    :return: a preprocessed copy of the environment
    """
    for preprocessor_config in preprocessor_configs:
        if "method" in preprocessor_config:
            try:
                preprocessor = getattr(env.unwrapped, preprocessor_config["method"])
                if "args" in preprocessor_config:
                    env = preprocessor(preprocessor_config["args"])
                else:
                    env = preprocessor()
            except AttributeError:
                logger.warning("The environment does not have a {} method".format(preprocessor_config["method"]))
        else:
            logger.error("The method is not specified in ", preprocessor_config)
    return env
