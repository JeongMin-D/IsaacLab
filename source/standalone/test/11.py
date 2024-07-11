import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
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
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

from omni.isaac.lab_assets.test import HAMSTER_N_CFG
from omni.isaac.lab_assets.test import ANYMAL_C_CFG
from omni.isaac.lab_assets.test import UNITREE_A1_CFG


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[2.5, 2.5, 1.0], [-2.5, 2.5, 1.0], [2.5, -2.5, 1.0], [-2.5, -2.5, 1.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    hamster_cfg = HAMSTER_N_CFG.copy()
    hamster_cfg.prim_path = "/World/Origin.*/Robot"
    hamster = Articulation(cfg=hamster_cfg)

    # anymal_cfg = ANYMAL_C_CFG.copy()
    # anymal_cfg.prim_path = "/World/Origin.*/Robot"
    # anymal = Articulation(cfg=anymal_cfg)

    # a1_cfg = UNITREE_A1_CFG.copy()
    # a1_cfg.prim_path = "/World/Origin.*/Robot"
    # a1 = Articulation(cfg=a1_cfg)

    # return the scene information
    scene_entities = {"hamster": hamster}

    # scene_entities = {"anymal": anymal}

    # scene_entities = {"a1": a1}

    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    hamster = entities["hamster"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset root state
            root_state = hamster.data.default_root_state.clone()
            # sample a random position on a cylinder around the origins
            root_state[:, :3] += origins
            root_state[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 0.5), size=hamster.num_instances, device=hamster.device
            )
            # write root state to simulation
            hamster.write_root_state_to_sim(root_state)
            # reset buffers
            hamster.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # apply sim data
        hamster.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        hamster.update(sim_dt)
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {hamster.data.root_state_w[:, :3]}")


# def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
#     """Runs the simulation loop."""
#     # Extract scene entities
#     # note: we only do this here for readability. In general, it is better to access the entities directly from
#     #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
#     anymal = entities["anymal"]
#     # Define simulation stepping
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0
#     # Simulate physics
#     while simulation_app.is_running():
#         # reset
#         if count % 250 == 0:
#             # reset counters
#             sim_time = 0.0
#             count = 0
#             # reset root state
#             root_state = anymal.data.default_root_state.clone()
#             # sample a random position on a cylinder around the origins
#             root_state[:, :3] += origins
#             root_state[:, :3] += math_utils.sample_cylinder(
#                 radius=0.1, h_range=(0.25, 0.5), size=anymal.num_instances, device=anymal.device
#             )
#             # write root state to simulation
#             anymal.write_root_state_to_sim(root_state)
#             # reset buffers
#             anymal.reset()
#             print("----------------------------------------")
#             print("[INFO]: Resetting object state...")
#         # apply sim data
#         anymal.write_data_to_sim()
#         # perform step
#         sim.step()
#         # update sim-time
#         sim_time += sim_dt
#         count += 1
#         # update buffers
#         anymal.update(sim_dt)
#         # print the root position
#         if count % 50 == 0:
#             print(f"Root position (in world): {anymal.data.root_state_w[:, :3]}")


# def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
#     """Runs the simulation loop."""
#     # Extract scene entities
#     # note: we only do this here for readability. In general, it is better to access the entities directly from
#     #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
#     a1 = entities["a1"]
#     # Define simulation stepping
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0
#     # Simulate physics
#     while simulation_app.is_running():
#         # reset
#         if count % 250 == 0:
#             # reset counters
#             sim_time = 0.0
#             count = 0
#             # reset root state
#             root_state = a1.data.default_root_state.clone()
#             # sample a random position on a cylinder around the origins
#             root_state[:, :3] += origins
#             root_state[:, :3] += math_utils.sample_cylinder(
#                 radius=0.1, h_range=(0.25, 0.5), size=a1.num_instances, device=a1.device
#             )
#             # write root state to simulation
#             a1.write_root_state_to_sim(root_state)
#             # reset buffers
#             a1.reset()
#             print("----------------------------------------")
#             print("[INFO]: Resetting object state...")
#         # apply sim data
#         a1.write_data_to_sim()
#         # perform step
#         sim.step()
#         # update sim-time
#         sim_time += sim_dt
#         count += 1
#         # update buffers
#         a1.update(sim_dt)
#         # print the root position
#         if count % 50 == 0:
#             print(f"Root position (in world): {a1.data.root_state_w[:, :3]}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
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