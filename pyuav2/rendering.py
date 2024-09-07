import numpy as np
import pybullet as pyb


class PerspectiveCamera:
    def __init__(
            self, 
            env: int,
            origin: np.ndarray, 
            lookat: np.ndarray,
            fov: float = 45.0,
            width: int = 640,
            height: int = 480,
            clip_near: float = 0.1,
            clip_far: float = 300,
            shadows: bool = False
            ) -> None:
        """Initialize a new perspective camera.

        Args:
            env (BaseEnvironment):     Pybullet client ID (see Aviary.__init__())
            origin (np.ndarray):       Location of camera
            lookat (np.ndarray):       Location to view / look at
            fov (float):               Field of view (default: 45 deg)
            width (float):             Width of output render (default: 640)
            height (float):            Height of output render (default: 480)
            clip_near (float):         Near clipping plane (default: 0.1m)
            clip_far (float):          Far clipping plane (default: 300m)
            shadows (bool):            Whether to render shadows (default: True)
        """
        # PyBullet physicsClientId (!)
        self._client_id = env._env._client

        self._fov = fov
        self._width = width
        self._height = height
        self._clip_far = clip_far
        self._clip_near = clip_near
        self._shadows = shadows
        self._origin = origin.tolist()
        self._lookat = lookat.tolist()
        self._upref = [0, 0, 1] # up-reference (!)

        self._view_matrix = pyb.computeViewMatrix(
            cameraEyePosition=self._origin, 
            cameraTargetPosition=self._lookat, 
            cameraUpVector=self._upref, 
            physicsClientId=self._client_id
        )

        self._proj_matrix = pyb.computeProjectionMatrixFOV(
            fov=self._fov, 
            aspect=self._width / self._height, 
            nearVal=self._clip_near, 
            farVal=self._clip_far, 
            physicsClientId=self._client_id
        )

    def get_image(self, mode: str = "rgba") -> np.ndarray:
        """Fetch current RGBA or depth image from camera. 

        Args:
            mode (str): "rgba" or "depth"

        Returns:
            np.ndarray: Image frame pf shape (height, width, channels), where
                        channels=1 for mode='depth', and channels=4 for mode='rgba'
        """
        width, height, rgba, depth, _ = pyb.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            shadow=self._shadows,
            lightDirection=[1, 1, 1],
            physicsClientId=self._client_id,
            renderer=pyb.ER_BULLET_HARDWARE_OPENGL,
            flags=pyb.ER_NO_SEGMENTATION_MASK 
            )
        
        if mode == 'rgba':
            img = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)
        elif mode == 'depth':
            img = np.array(depth, dtype=np.float32).reshape(height, width, 1)
        else:
            raise Exception(f"Mode '{mode}' not understood.")

        return img
