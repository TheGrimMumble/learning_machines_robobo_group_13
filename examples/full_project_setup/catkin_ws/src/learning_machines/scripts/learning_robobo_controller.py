#!/usr/bin/env python3
import sys
import multiprocessing
from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.EC.task3_align import run_all_actions


def run(port_offset: int) -> None:
    """Worker function that runs on a specific CoppeliaSim instance"""
    rob = SimulationRobobo(api_port=(20000 + port_offset))
    run_all_actions(rob)


if __name__ == "__main__":
    # Check if we're running in parallel mode
    if len(sys.argv) >= 2 and sys.argv[1] == "--parallel":
        # Use the existing run_all_actions but with parallel support
        rob = None  # We'll handle robot creation in the evolutionary algorithm
        run_all_actions(rob, parallel=True, num_processes=30)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--file_path":
        if len(sys.argv) < 3:
            raise ValueError("Please provide the file path after --file_path argument.")

        # get whether it is hardware or simulation
        # if sys.argv[3] not in ["--hardware", "--simulation"]:
        #     raise ValueError(
        #         "Please specify --hardware or --simulation after --file_path."
        #     )
        # if sys.argv[2] == "--hardware":
        #     is_hardware = True
        # elif sys.argv[2] == "--simulation":
        #     is_hardware = False
        is_hardware = False

        file_path = sys.argv[2]
        rob = None
        print(f"Running with file path: {file_path}")
        run_all_actions(rob, file_path=file_path, is_hardware=is_hardware)
    elif len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware or simulation
             Pass `--hardware`, `--simulation`, or `--parallel` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
        run_all_actions(rob)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        run_all_actions(rob)
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
