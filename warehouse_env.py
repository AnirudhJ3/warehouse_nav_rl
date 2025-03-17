import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from pathlib import Path
import os

class WarehouseEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),  # [linear velocity, angular velocity]
            dtype=np.float32
        )
        
        # Observation space: [robot_pos_x, robot_pos_y, robot_orientation, 
        #                    goal_pos_x, goal_pos_y, lidar_readings]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),  # 3 robot state + 2 goal + 18 lidar readings
            dtype=np.float32
        )
        
        # Load MuJoCo model
        model_path = os.path.join(Path(__file__).parent, "assets", "warehouse.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.max_episode_steps = 500
        self.current_step = 0
        self.render_mode = render_mode
        
        # Robot parameters
        self.robot_radius = 0.3
        self.goal_threshold = 0.5
        self.collision_threshold = 0.2
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial robot position and goal
        self.robot_pos = self._get_random_free_position()
        self.goal_pos = self._get_random_free_position()
        
        # Set initial state
        self.data.qpos[:2] = self.robot_pos
        self.data.qpos[2] = np.random.uniform(-np.pi, np.pi)
        
        self.current_step = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        
        # Apply action
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        # Get current state
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check termination conditions
        done = self._is_done()
        truncated = self.current_step >= self.max_episode_steps
        
        info = {
            'is_success': self._is_success(),
            'distance_to_goal': self._get_distance_to_goal()
        }
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self):
        robot_state = np.concatenate([
            self.data.qpos[:2],  # position
            [np.cos(self.data.qpos[2]), np.sin(self.data.qpos[2])]  # orientation
        ])
        
        # Simulate lidar readings
        lidar_readings = self._get_lidar_readings()
        
        return np.concatenate([
            robot_state,
            self.goal_pos,
            lidar_readings
        ])
    
    def _get_lidar_readings(self):
        # Simulate 18 lidar readings (every 20 degrees)
        num_readings = 18
        readings = np.zeros(num_readings)
        
        for i in range(num_readings):
            angle = i * 2 * np.pi / num_readings
            readings[i] = self._ray_cast(angle)
        
        return readings
    
    def _ray_cast(self, angle):
        # Simplified ray casting implementation
        # In real implementation, use MuJoCo's collision detection
        max_distance = 5.0
        return max_distance
    
    def _compute_reward(self):
        distance_to_goal = self._get_distance_to_goal()
        collision_penalty = self._check_collision() * -10.0
        
        # Reward shaping
        reward = -distance_to_goal  # Dense reward
        reward += collision_penalty
        
        if self._is_success():
            reward += 100.0  # Bonus for reaching goal
            
        return reward
    
    def _is_success(self):
        return self._get_distance_to_goal() < self.goal_threshold
    
    def _is_done(self):
        return self._is_success() or self._check_collision()
    
    def _get_distance_to_goal(self):
        robot_pos = self.data.qpos[:2]
        return np.linalg.norm(robot_pos - self.goal_pos)
    
    def _check_collision(self):
        # Implement collision checking with obstacles
        # For now, return False (no collision)
        return False
    
    def _get_random_free_position(self):
        # Generate random position in the warehouse
        # Ensure it's not colliding with obstacles
        return np.random.uniform(-5, 5, size=2)
    
    def render(self):
        if self.render_mode == "human":
            # Implement rendering using MuJoCo's viewer
            pass
        
    def close(self):
        if hasattr(self, 'viewer'):
            self.viewer.close() 