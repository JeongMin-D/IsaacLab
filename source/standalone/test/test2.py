# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage with terrain in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/create_terrain.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating a stage with terrain.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.terrains import TerrainGenerator, TerrainGeneratorCfg, SubTerrainBaseCfg
import omni.usd

# Define a terrain generation function
def perlin_noise_terrain(difficulty, cfg):
    # Here we would define the actual terrain generation logic based on Perlin noise
    # For demonstration purposes, we will just return mock data
    size = cfg.size
    mesh = None  # Replace with actual mesh generation logic
    origin = (0, 0, 0)  # Replace with actual origin calculation logic
    return mesh, origin

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Stage 설정
    stage = omni.usd.get_context().get_stage()
    omni.isaac.core.utils.prims.create_prim('/World', 'Xform')

    # SubTerrainBaseCfg 설정
    sub_terrain_cfg = SubTerrainBaseCfg(
        function=perlin_noise_terrain,  # 함수 유형
        proportion=1.0,  # 비율
        size=(10, 10),  # 타일 수 (X, Y)
        flat_patch_sampling=False  # 평탄한 패치 샘플링 사용 여부
    )

    # TerrainGeneratorCfg 설정
    terrain_cfg = TerrainGeneratorCfg(
        seed=42,  # 랜덤 시드 값
        size=(10, 10),  # 타일 수 (X, Y)
        num_rows=1,
        num_cols=1,
        border_width=0,
        sub_terrains={"terrain_0": sub_terrain_cfg},  # SubTerrainBaseCfg 딕셔너리
        horizontal_scale=1.0,  # 수평 스케일
        vertical_scale=1.0,  # 수직 스케일
        slope_threshold=1.0,  # 경사 임계값
        use_cache=False  # 캐시 사용 여부
    )

    # TerrainGenerator 초기화 및 지형 생성
    terrain_generator = TerrainGenerator(cfg=terrain_cfg)
    terrain_generator.generate(stage, '/World/Terrain')

    # 지형의 첫 타일의 위치 확인
    first_tile_prim = stage.GetPrimAtPath('/World/Terrain/tile_0_0')
    print(f"First tile prim path: {first_tile_prim.GetPath()}")

    # 업데이트 및 뷰어 설정
    omni.usd.get_context().save_as_stage("/path/to/your/saved_terrain.usd")  # 지형을 USD 파일로 저장
    omni.kit.commands.execute('ChangeProperty', prop_path=Sdf.Path('/World/Terrain'), value=(0.0, 0.0, 0.0))  # 지형의 위치 설정

    print("[INFO]: Terrain generation complete and saved to /path/to/your/saved_terrain.usd")

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
