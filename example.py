"""Spawns three drones, then sets all drones to have different control looprates."""
import cv2
import numpy as np
from pyuav2.rendering import PerspectiveCamera
from pyuav2.environments import Environment

# Starting positions of 4 drones
start_pos = np.array([
    [-19.0, 10.0, 1.0], 
    [-20.0, 11.0, 1.0], 
    [-18.0, 13.0, 1.0],
    [-19.0, 13.0, 1.0],
    ])

# Create hangar environment with Boeing 787
hangar = Environment(
    num_drones=4,
    start_pos=start_pos,
    start_rot=np.zeros_like(start_pos), # start perfectly level (!)
    control_mode="positional"
)
hangar.add_obstacle(
    path_to_obj="assets/boeing-787/boeing-787.obj",
    position=[0, 0, 0]
)

# Create camera to render outside view
camera = PerspectiveCamera(
    env=hangar, 
    origin=np.array([-31, 17, 5]),
    lookat=np.mean(start_pos, axis=0) + np.array([0, -3, 3])
    )

# Simulate!
states = hangar.get_states()

for i in range(1000):

    # Control drones to all fly to same location facing north!
    setpoint = [(x, y, 0, z + 5) for x, y, z in start_pos]
    hangar.set_all_setpoints(setpoint)

    # Advance simulation state
    states = hangar.step()

    # Visualize
    if i % 5 == 0:
        rgb = camera.get_image()

        cv2.imshow('', rgb)
        cv2.waitKey(1)