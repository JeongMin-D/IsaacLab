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

import os

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from omni.isaac.lab.terrains.terrain_generator import TerrainGenerator
from omni.isaac.lab.terrains.terrain_importer import TerrainImporter

##
# Pre-defined configs
##
from asset.quadruped import HAMSTER_N_CFG


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    #cfg = sim_utils.GroundPlaneCfg()
    #cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "generator")

    test_terrain_cfg = TerrainGeneratorCfg(
        size = [10.0, 10.0],
        border_width = 0.0,
        num_rows = 1,
        num_cols = 1,
        horizontal_scale = 0.1,
        vertical_scale = 0.005,
        slope_threshold = 0.75,
        use_cache = True,
        curriculum = True,
        cache_dir = test_dir,
        sub_terrains={
            "pyramid_stairs":terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion = 0.3,
                step_height_range = (0.05, 0.23),
                step_width = 0.3,
                platform_width = 3.0,
                border_width = 1.0,
                holes = False,
            ),
            # "pyramid_stairs_inv":terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            #     proportion = 0.3,
            #     step_height_range = (0.05, 0.23),
            #     step_width = 0.3,
            #     platform_width = 3.0,
            #     border_width = 1.0,
            #     holes = False,
            # ),
            # "random_uniform":terrain_gen.HfRandomUniformTerrainCfg(
            #     proportion = 0.3,
            #     noise_range = (0.1, 0.6),
            #     noise_step = 0.05,
            #     #size = (10.0, 10.0),
            #     border_width = 1.0,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            # ),
            # "pyramid_slope":terrain_gen.HfPyramidSlopedTerrainCfg(
            #     proportion = 0.3,
            #     slope_range = (0.05, 0.5),
            #     platform_width = 0.3,
            #     inverted = False,
            #     border_width = 0.3,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            #     slope_threshold = 0.75,
            # ),
            # "pyramid_slope_inv":terrain_gen.HfPyramidSlopedTerrainCfg(
            #     proportion = 0.3,
            #     slope_range = (0.05, 0.5),
            #     platform_width = 0.3,
            #     inverted = True,
            #     border_width = 0.3,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            #     slope_threshold = 0.75,
            # ),
            "hf_pyramid_stairs_slope":terrain_gen.HfPyramidStairsTerrainCfg(
                proportion = 1.0,
                step_height_range = (0.3, 0.3),
                step_width = 0.5,
                platform_width = 0.1,
                inverted = False,
                border_width = 0.0,
                horizontal_scale = 0.1,
                vertical_scale = 0.005,
                slope_threshold = 0.75,
            ),
            # "hf_pyramid_stairs_slope_inv":terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            #     proportion = 0.3,
            #     step_height_range = (0.05, 0.3),
            #     step_width = 0.3,
            #     platform_width = 0.1,
            #     inverted = True,
            #     border_width = 0.3,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            #     slope_threshold = 0.75,
            # ),
            # "discrete_obstacles":terrain_gen.HfDiscreteObstaclesTerrainCfg(
            #     proportion = 0.3,
            #     obstacle_height_mode = "choice",
            #     border_width = 0.0,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            #     slope_threshold = None,
            #     obstacle_width_range = (0.1, 0.3),
            #     obstacle_height_range = (0.1, 0.3),
            #     num_obstacles = 5,
            # ),
            # "wave":terrain_gen.HfWaveTerrainCfg(
            #     proportion = 1.0,
            #     border_width = 0.0,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            #     amplitude_range = (0.5, 0.5),
            #     num_waves = 3,
            # ),
            # "stepping_stones":terrain_gen.HfSteppingStonesTerrainCfg(
            #     proportion = 0.3,
            #     border_width = 0.0,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            #     slope_threshold = None,
            #     stone_height_max = 0.3,
            #     stone_width_range = (0.1, 0.3),
            #     stone_distance_range = (0.1, 0.3),
            #     holes_depth = -10.0,
            #     platform_width = 1.0,
            # ),
            # # border_width must be greater than 0(border_width = cfg.size[0] - min(num_boxes_x, num_boxes_y) * cfg.grid_width)
            # "random_grid":terrain_gen.MeshRandomGridTerrainCfg(
            #     proportion = 1.0,
            #     grid_width = 10.0,
            #     grid_height_range = (0.1, 1.0),
            #     platform_width = 0.3,
            #     holes = True,
            # ),
            # "rails":terrain_gen.MeshRailsTerrainCfg(
            #     proportion = 0.3,
            #     rail_thickness_range = (0.1, 0.3),
            #     rail_height_range = (0.1, 0.3),
            #     platform_width = 1.0,
            # ),
            # "pit":terrain_gen.MeshPitTerrainCfg(
            #     proportion = 0.3,
            #     pit_depth_range = (0.1, 0.3),
            #     platform_width = 1.0,
            #     double_pit = False,
            # ),
            # "box":terrain_gen.MeshBoxTerrainCfg(
            #     proportion = 1.0,
            #     box_height_range = (1.0, 1.0),
            #     platform_width = 5.0,
            #     double_box = False,
            # ),
            # "gap":terrain_gen.MeshGapTerrainCfg(
            #     proportion = 0.3,
            #     gap_width_range = (0.1, 0.3),
            #     platform_width = 1.0,
            # ),
            # "floating_ring":terrain_gen.MeshFloatingRingTerrainCfg(
            #     proportion = 0.3,
            #     ring_width_range = (0.1, 0.3),
            #     ring_height_range = (0.1, 0.3),
            #     ring_thickness = 0.1,
            #     platform_width = 1.0,
            # ),
            # "star":terrain_gen.MeshStarTerrainCfg(
            #     proportion = 0.3,
            #     num_bars = 2,
            #     bar_width_range = (0.1, 0.3),
            #     bar_height_range = (0.1, 0.3),
            #     platform_width = 1.0,
            # ),
        }
    )

    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        num_envs = 10,
        #env_spacing = 10.0,
        prim_path = "/World/Terrain",
        max_init_terrain_level = None,
        terrain_type = "generator",
        visual_material = None,
        terrain_generator = test_terrain_cfg.replace(curriculum=True, color_scheme="height"),
        physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode = "multiply",
            restitution_combine_mode = "multiply",
            static_friction = 1.0,
            dynamic_friction = 1.0,
        ),
    )

    # terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
    #     prim_path = "/World/Terrain",
    #     num_envs = 1,
    #     terrain_type = "usd",
    #     usd_path = "/home/jmin/isaac_ws/box.usd",
    #     env_spacing = 0.0,
    # )

    terrain_importer = TerrainImporter(terrain_importer_cfg)
    #terrain_importer = TerrainImporter(terrain_importer_cfg).import_usd(key = "test", usd_path = "/home/jmin/isaac_ws/box.usd")
    

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[100.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation
    hamster_n_cfg = HAMSTER_N_CFG.copy()
    hamster_n_cfg.prim_path = "/World/Origin.*/Robot"
    hamster_n = Articulation(cfg=hamster_n_cfg)

    # return the scene information
    scene_entities = {"hamster_n": hamster_n}
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