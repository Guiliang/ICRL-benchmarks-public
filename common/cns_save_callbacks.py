import os
from typing import Optional

import gym

from stable_baselines3.common import vec_env
from stable_baselines3.common.callbacks import EventCallback, BaseCallback


class CNSCheckpointCallback(EventCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param callback_on_new_save: callback to trigger when model is saved
    """

    def __init__(
            self,
            env: gym.Env,
            save_freq: int,
            save_path: str,
            name_prefix="model",
            verbose=0,
            callback_on_new_save: Optional[BaseCallback] = None
        ):
        super(CNSCheckpointCallback, self).__init__(callback_on_new_save, verbose)
        self.env = env
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if not os.path.exists(os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")):
                os.mkdir(os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps"))
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps", 'gail_net')
            self.model.save(path)
            if isinstance(self.env, vec_env.VecNormalize):
                self.env.save(os.path.join(self.save_path,
                                           f"{self.name_prefix}_{self.num_timesteps}_steps",
                                           "train_env_stats.pkl"))
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
            if self.callback is not None:
                self._on_event()
        return True