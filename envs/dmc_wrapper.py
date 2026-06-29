import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dm_control import suite
from dm_env import specs

def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype in [np.float64, np.float32]
        dim = int(np.prod(s.shape)) # Fixed np.int deprecation
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    return spaces.Box(low, high, dtype=dtype)

def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)

class DMCGym(gym.Env):
    def __init__(
        self,
        domain,
        task,
        task_kwargs=None,
        environment_kwargs=None,
        from_pixels=True,
        normalize_actions=True,
        height=64,
        width=64,
        camera_id=0,
        frame_skip=4,
        channels_first=True,
        rendering="egl",
        terminate_on_limit=True,
    ):
        os.environ["MUJOCO_GL"] = rendering
        task_kwargs = task_kwargs or {}
        
        self._env = suite.load(
            domain, task, task_kwargs, environment_kwargs
        )

        self._domain = domain
        self._from_pixels = from_pixels
        self._normalize_actions = normalize_actions
        self._terminate_on_limit = terminate_on_limit
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        # Action Spaces: True (MuJoCo) vs Normalized (Agent)
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        if self._normalize_actions:
            self._action_space = spaces.Box(
                low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
            )
        else:
            # If disabled, the agent sees the raw bounds (e.g., -150 to 150 for some motors)
            self._action_space = self._true_action_space

        # Observation Space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values(), np.float32
            )

        # Separate State Space (always flattened state)
        self._state_space = _spec_to_box(
            self._env.observation_spec().values(), np.float32
        )
        
        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, timestep):
        if self._from_pixels:
            obs = self.render(
                height=self._height, width=self._width, camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(timestep.observation)
        return obs

    def _convert_action(self, action):
        if not self._normalize_actions:
            return action.astype(np.float32)
        action = action.astype(np.float32)
        true_low = self._true_action_space.low
        true_high = self._true_action_space.high
        # Map from [-1, 1] to [true_low, true_high]
        action = true_low + (action + 1.0) * 0.5 * (true_high - true_low)
        action = np.clip(action, true_low, true_high)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def step(self, action):
        # 1. Normalize action
        action = self._convert_action(action)
        
        # 2. Frame skip and reward accumulation
        total_reward = 0.0
        terminated = False
        for _ in range(self._frame_skip):
            timestep = self._env.step(action)
            total_reward += (timestep.reward or 0.0)
            if self._terminate_on_limit and self._domain == 'cartpole' and abs(self._env.physics.cart_position()) >= 1.8:
                terminated = True
            if timestep.last() or terminated:
                break
        
        # 3. Process Observations
        obs = self._get_obs(timestep)
        self.current_state = _flatten_obs(timestep.observation)
        
        # 4. Gymnasium returns
        truncated = timestep.last() and not terminated
        info = {
            "internal_state": self._env.physics.get_state().copy(),
            "discount": timestep.discount,
            "observation_state": self.current_state,
            "terminated": terminated,
            "truncated": truncated
        }
        
        return obs, total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # DMC handles seeds via task_kwargs, but we can update it here
        if seed is not None:
            self._env.task._random = np.random.RandomState(seed)
        
        timestep = self._env.reset()
        self.current_state = _flatten_obs(timestep.observation)
        obs = self._get_obs(timestep)
        
        return obs, {}

    def render(self, height=None, width=None, camera_id=None):
        return self._env.physics.render(
            height=height or self._height,
            width=width or self._width,
            camera_id=camera_id or self._camera_id,
        )