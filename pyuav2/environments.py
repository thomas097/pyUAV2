import numpy as np
from PyFlyt.core import Aviary, loadOBJ, obj_collision, obj_visual


class Environment:

    DRONE_MODELS = [
        "primitive_drone", 
        "cf2x"
        ]
    
    CONTROL_MODES = {
        "positional": 7, # [x, y, yaw, z]
        "velocity": 6    # [vx, vy, vr, vz]
    }

    def __init__(
            self,
            num_drones: int,
            start_pos: np.ndarray,
            start_rot: np.ndarray,
            drone_model: str = "primitive_drone",
            control_mode: str = "positional",
            use_camera: bool = False,
            physics_hz: int = 240
        ) -> None:
        """Initializes an empty environment with specified number of quadcopter UAVs.

        Args:
            num_drones (int):             Number of drones to spawn in environment.
            start_pos (np.ndarray):       An Nx3 matrix of starting positions in carthesian XYZ coordinates.
            start_rot (np.ndarray):       An Nx3 matrix of starting orientations in XYZ euler angles.
            drone_model (str, optional):  Drone model to use for each drone:
                                           - "primitive_drone": model of idealized drone dynamics (default).
                                           - "cf2x": tuned through system identification to mimic a 
                                              Crazyflie 2.1 quadcopter.
            control_mode (str, optional): Type of control input (i.e., setpoints) to use:
                                           - "positional" - using control inputs [x, y, yaw, z] (default).
                                           - "velocity" - using control inputs [vx, vy, vr, vz].
            use_camera (bool, optional):  Whether to output camera images from the onboard cameras of
                                          each UAV (default: False).
            physics_hz (int, optional):   Physics update rate (default: 240 hertz). Changing `physics_hz`
                                          is discouraged as this may result in unexpected physics behavior.
        """

        assert control_mode in self.CONTROL_MODES, \
            f"control_mode '{control_mode}' not understood; choose from {list(self.CONTROL_MODES.keys())}"
        assert drone_model in self.DRONE_MODELS, \
            f"drone_model '{drone_model}' not understood; choose from {self.DRONE_MODELS}"
        assert num_drones == start_pos.shape[0] == start_rot.shape[0], \
            f"Number of positions and rotations specified must be equal to num_drones."

        # Setup PyFlyt environment
        drone_options = dict(
            use_camera=use_camera,
            drone_model=drone_model
            )
        
        self._env = Aviary(
            start_pos=start_pos,
            start_orn=start_rot,
            render=False,
            drone_type="quadx",
            physics_hz=physics_hz,
            drone_options=drone_options
        )

        self._env.set_mode(self.CONTROL_MODES[control_mode])

    def add_obstacle(
            self,
            path_to_obj: str,
            position: list | np.ndarray = [0, 0, 0],
            mass: float = 0,
            concave: bool = False
            ) -> None:
        """Add an obstacle to the environment.

        Args:
            path_to_obj (str):         Path to mesh file in wavefront obj format.
            position (list | ndarray): Position in environment (default: [0, 0, 0]).
            mass (float):              Mass of obstacle. If static, set mass=0 (default: 0).
            concave (bool):            Whether the obstacle is concave (default: False).
        """
        visualId = obj_visual(self._env, path_to_obj)
        collisionId = obj_collision(self._env, path_to_obj, concave=concave)
        
        loadOBJ(
            self._env,
            visualId=visualId,
            collisionId=collisionId,
            baseMass=mass,
            basePosition=position,
        )

        self._env.register_all_new_bodies()

    def set_all_setpoints(self, setpoints: np.ndarray) -> None:
        """Sets the setpoints of each drone in the environment.

        Args:
            setpoints (np.ndarray): list of setpoints; one for each UAV.
        """
        assert len(setpoints) == len(self._env.drones)
        self._env.set_all_setpoints(np.array(setpoints, dtype=np.float32))

    def set_setpoint(self, idx: int, setpoint: np.ndarray) -> None:
        """Sets the setpoint of a single UAV, designated by `idx`.

        Args:
            idx (int):             ID designating drone to assign setpoint to, where 0 <= idx < N
            setpoint (np.ndarray): Setpoint for given UAV.
        """
        assert 0 <= idx < len(self._env.drones)
        self._env.set_setpoint(index=idx, setpoint=np.array(setpoint, dtype=np.float32))

    def get_states(self) -> list:
        return self._env.all_states

    def step(self) -> None:
        """Advance simulation state.
        """
        self._env.step()
        return self._env.all_states

    def get_camera_images(self, mode: str = "rgba") -> list[np.ndarray]:
        """Fetch current RGBA or depth images from onboard cameras of drones. 

        Args:
            mode (str): "rgba" or "depth"

        Returns:
            list[ndarray]: Sequence of frames of shape (height, width, channels), where
                           channels=1 for mode='depth', and channels=4 for mode='rgba'
        """
        imgs = []
        for drone in self._env.drones:
            if mode == 'rgba':
                imgs.append(drone.rgbaImg)
            elif mode == 'depth':
                imgs.append(drone.depthImg)
            else:
                raise Exception(f"Mode '{mode}' not understood.")

        return imgs

    