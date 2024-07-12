import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

from omni.isaac.lab.terrains import TerrainGenerator ,TerrainGeneratorCfg, SubTerrainBaseCfg, TerrainImporterCfg, TerrainImporter
from omni.isaac.lab.terrains.height_field.hf_terrains import random_uniform_terrain, pyramid_sloped_terrain
from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg

##
# Pre-defined configs
##
from asset.quadruped import HAMSTER_N_CFG


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)


    hf_pyramid_sloped_terrain_cfg = HfPyramidSlopedTerrainCfg(
        slope_range=(0.0, 1.0),
        platform_width = 1.0,
        inverted = False,
        proportion = 1.0,
        size = (10.0, 10.0),
        border_width = 0.5,
        horizontal_scale = 0.1,
        vertical_scale = 0.005,
        slope_threshold = 0.75,
    )

    hf_pyramid_sloped_terrain = pyramid_sloped_terrain(
        difficulty = 0.3,
        cfg = hf_pyramid_sloped_terrain_cfg
    )

    sub_terrain_1_cfg = SubTerrainBaseCfg(
        function= hf_pyramid_sloped_terrain,
        proportion=1.0,
        size=(10.0, 10.0),
    )

    sub_terrain = {"sub_terrain_1": sub_terrain_1_cfg}

    print("sub_terrain:", type(sub_terrain))

    terrain_cfg = TerrainGeneratorCfg(
        seed = 1,
        curriculum = False,
        size = (10.0, 10.0),
        border_width = 0.5,
        num_rows = 1,
        num_cols = 1,
        color_scheme = "height",
        horizontal_scale = 0.1,
        vertical_scale = 0.005,
        slope_threshold = 0.75,
        sub_terrains = sub_terrain,
        difficulty_range = (0.0, 1.0),
        #use_cache = True,
        #cache_dir = "/home/jmin/isaac_ws/IsaacLab/docs/source/_static/terrain/height_field"
    )
    terrain_generator = TerrainGenerator(cfg=terrain_cfg, device="cuda:0")

    terrain_import_cfg = TerrainImporterCfg(
        prim_path="/World/Terrain",
        terrain_type="generator",
        terrain_generator=terrain_generator,
    )

    terrain_importer = TerrainImporter(cfg=terrain_import_cfg)


    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation
    hamster_n_cfg = HAMSTER_N_CFG.copy()
    hamster_n_cfg.prim_path = "/World/Origin.*/Robot"
    hamster_n = Articulation(cfg=hamster_n_cfg)

    # return the scene information
    scene_entities = {"hamster_n": hamster_n, "terrain": terrain_importer}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["hamster_n"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Apply random action
        # -- generate random joint efforts
        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()

        # -- apply action to the robot
        test_pos = torch.tensor(robot.data.default_joint_pos, dtype=torch.float)
        test_vel = torch.tensor([[0]*12], dtype=torch.float)
        test_effort = torch.tensor([[0]*12], dtype=torch.float)

        robot.set_joint_position_target(test_pos)
        robot.set_joint_velocity_target(test_vel)
        robot.set_joint_effort_target(test_effort)

        print("joint_pos:", joint_pos)
        print("joint_vel", joint_vel)

        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim_cfg.dt = 0.005
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()