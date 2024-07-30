# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Hamster-N-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.HamsterNFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HamsterNFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Hamster-N-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.HamsterNFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HamsterNFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Hamster-N-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.HamsterNRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HamsterNRoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Hamster-N-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.HamsterNRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HamsterNRoughPPORunnerCfg,
    },
)
