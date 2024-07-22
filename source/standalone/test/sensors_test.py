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

from omni.isaac.lab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, Camera, ContactSensor, TiledCameraCfg, TiledCamera

##
# Pre-defined configs
##
from asset.quadruped import HAMSTER_N_CFG

def define_sensor():
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin.*/Robot/base",
        mesh_prim_paths=["/World/Terrain"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
        attach_yaw_only=True,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        debug_vis=not args_cli.headless,
        #debug_vis=False,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    camera_cfg = CameraCfg(
        prim_path="/World/Origin.*/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.510, 0.0, 0.015),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros"
        ),
    )
    camera = Camera(cfg=camera_cfg)

    contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/Origin.*/Robot/.*_FOOT",
        update_period=0.0,
        history_length=6,
        debug_vis=not args_cli.headless,
    )
    contact_sensor = ContactSensor(cfg=contact_sensor_cfg)

    tiled_camera_cfg = TiledCameraCfg(
        prim_path="/World/Origin.*/Robot/base/front_camera_depth_optical_frame",
        update_period=0.0,
        history_length=1.0,
        debug_vis=not args_cli.headless,
        data_types=["rgb", "depth"],
        width=640,
        height=480,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0, 1.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros"
        ),
    )
    tiled_camera = TiledCamera(cfg=tiled_camera_cfg)

    return ray_caster, camera, contact_sensor, tiled_camera


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
        size = [5.0, 5.0],
        border_width = 0.0,
        num_rows = 1,
        num_cols = 1,
        horizontal_scale = 0.1,
        vertical_scale = 0.005,
        slope_threshold = 0.75,
        use_cache = True,
        curriculum = False,
        cache_dir = test_dir,
        sub_terrains={
            # "box":terrain_gen.MeshBoxTerrainCfg(
            #     proportion = 0.3,
            #     box_height_range = (0.5, 0.5),
            #     platform_width = 1.0,
            #     double_box = False,
            # ),
            # "pyramid_stairs":terrain_gen.MeshPyramidStairsTerrainCfg(
            #     proportion = 0.4,
            #     step_height_range = (0.2, 0.2),
            #     step_width = 0.3,
            #     platform_width = 1.0,
            #     border_width = 0.5,
            #     holes = True,
            # ),
            "pyramid_stairs_inv":terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion = 1.0,
                step_height_range = (0.05, 0.23),
                step_width = 0.3,
                platform_width = 3.0,
                border_width = 1.0,
                holes = False,
            ),
            # "pyramid_stairs":terrain_gen.HfPyramidStairsTerrainCfg(
            #     proportion = 0.4,
            #     step_height_range = (0.2, 0.2),
            #     step_width = 0.3,
            #     platform_width = 1.0,
            #     border_width = 0.5,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            #     slope_threshold = 0.75,
            # ),
            # "random_uniform":terrain_gen.HfWaveTerrainCfg(
            #     proportion = 0.3,
            #     border_width = 0.1,
            #     horizontal_scale = 0.1,
            #     vertical_scale = 0.005,
            #     slope_threshold = 0.75,
            #     amplitude_range = (0.0, 1.0),
            #     num_waves = 3,
            # ),
        }
    )

    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        num_envs = 10,
        env_spacing = 10.0,
        prim_path = "/World/Terrain",
        max_init_terrain_level = None,
        terrain_type = "generator",
        visual_material = None,
        terrain_generator = test_terrain_cfg.replace(curriculum=True, color_scheme="height"),
    )

    terrain_importer = TerrainImporter(terrain_importer_cfg)

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

    ray_caster, camera, contact_sensor, tiled_camera = define_sensor()

    # return the scene information
    scene_entities = {"hamster_n": hamster_n, "ray_caster": ray_caster, "camera": camera, "contact_sensor": contact_sensor, "tiled_camera": tiled_camera}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["hamster_n"]
    ray_caster = entities["ray_caster"]
    camera = entities["camera"]
    contact_sensor = entities["contact_sensor"]
    tiled_camera = entities["tiled_camera"]
    #ray_caster = entities["ray_caster"]
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

        #print("joint_pos:", joint_pos)
        #print("joint_vel", joint_vel)
        
        
        # print information from the sensors
        # print("-------------------------------")
        # print(camera)
        # print("Received shape of rgb   image: ", camera.data.output["rgb"].shape)
        # print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        # print("-------------------------------")
        # print(ray_caster)
        # print("Received max height value: ", torch.max(ray_caster.data.ray_hits_w[..., -1]).item())
        # print("-------------------------------")
        # print(contact_sensor)
        # print("Received max contact force of: ", torch.max(contact_sensor.data.net_forces_w).item())
        # print("-------------------------------")
        print(tiled_camera)
        #print("Received shape of rgb   image: ", tiled_camera.data.output["rgb"].shape)
        print("Received shape of depth image: ", tiled_camera.data.output["depth"].shape)

        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)
        ray_caster.update(dt=sim_dt, force_recompute=True)
        camera.update(dt=sim_dt)
        contact_sensor.update(dt=sim_dt)


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