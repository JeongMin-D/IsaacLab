# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen

from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion = 1.0,
            step_height_range = (0.23, 0.5),
            step_width = 0.5,
            platform_width = 3.0,
            border_width = 1.0,
            holes = False,
        ),
        "box":terrain_gen.MeshBoxTerrainCfg(
            proportion = 1.0,
            box_height_range = (0.3, 0.3),
            platform_width = 3.0,
            double_box = False,
        ),
        "hf_pyramid_stairs_slope":terrain_gen.HfPyramidStairsTerrainCfg(
            proportion = 1.0,
            step_height_range = (0.3, 0.3),
            step_width = 0.5,
            platform_width = 0.3,
            inverted = False,
            border_width = 0.0,
            horizontal_scale = 0.1,
            vertical_scale = 0.005,
            slope_threshold = 0.75,
        ),
        "wave":terrain_gen.HfWaveTerrainCfg(
            proportion = 1.0,
            border_width = 0.0,
            horizontal_scale = 0.1,
            vertical_scale = 0.005,
            amplitude_range = (0.5, 0.5),
            num_waves = 1,
        ),
    },
)
"""Rough terrains configuration."""
