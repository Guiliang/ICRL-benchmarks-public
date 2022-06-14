from gym.envs.registration import register

ABS_PATH = "custom_envs.envs"

# =========================================================================== #
#                              Inverted Pendulum                              #
# =========================================================================== #

PENDULUM_LEN = 100

register(
    id="InvertedPendulumWall-v0",
    entry_point=ABS_PATH+".inverted_pendulum:InvertedPendulumWall",
    max_episode_steps=PENDULUM_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                                   Cheetah                                   #
# =========================================================================== #

CHEETAH_LEN = 1000

register(
    id="HCWithPos-v0",
    entry_point=ABS_PATH+".half_cheetah:HalfCheetahWithPos",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                                   Walker                                    #
# =========================================================================== #

WALKER_LEN = 500

register(
    id="WalkerWithPos-v0",
    entry_point=ABS_PATH+".walker:WalkerWithPos",
    max_episode_steps=WALKER_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                                  Swimmer                                    #
# =========================================================================== #

SWIMMER_LEN = 500

register(
    id="SwimmerWithPos-v0",
    entry_point=ABS_PATH+".swimmer:SwimmerWithPos",
    max_episode_steps=SWIMMER_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                                   Ant                                       #
# =========================================================================== #

ANT_LEN = 500

register(
    id="AntWall-v0",
    entry_point=ABS_PATH+".ant:AntWall",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                               Wall Grid World                               #
# =========================================================================== #

WALL_GRID_WORLD_LEN = 50  # depends on the settings in './configs/'

register(
    id="WGW-v0",
    entry_point=ABS_PATH+".wall_gird_word:WallGridworld",
    max_episode_steps=WALL_GRID_WORLD_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

