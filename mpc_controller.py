import numpy as np
from scipy.optimize import minimize
import cvxopt
from cvxopt import matrix, solvers
import time

class MPCController:
    def __init__(self, horizon=10, dt=0.1):
        self.horizon = horizon  # Prediction horizon
        self.dt = dt  # Time step
        
        # State and control constraints
        self.max_velocity = 1.0
        self.max_angular_velocity = np.pi/2
        self.max_acceleration = 2.0
        self.max_angular_acceleration = np.pi
        
        # Cost weights
        self.Q = np.diag([1.0, 1.0, 0.5])  # State cost
        self.R = np.diag([0.1, 0.1])  # Control cost
        self.P = np.diag([10.0, 10.0, 5.0])  # Terminal cost
        
        # Initialize solver
        solvers.options['show_progress'] = False
        
    def predict_state(self, state, control, dt):
        """Predict next state using unicycle model."""
        x, y, theta = state
        v, omega = control
        
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + omega * dt
        
        return np.array([x_next, y_next, theta_next])
    
    def get_linearized_dynamics(self, state, control):
        """Get linearized dynamics matrices A and B."""
        x, y, theta = state
        v, _ = control
        
        # Linearized A matrix
        A = np.array([
            [1, 0, -v * np.sin(theta) * self.dt],
            [0, 1, v * np.cos(theta) * self.dt],
            [0, 0, 1]
        ])
        
        # Linearized B matrix
        B = np.array([
            [np.cos(theta) * self.dt, 0],
            [np.sin(theta) * self.dt, 0],
            [0, self.dt]
        ])
        
        return A, B
    
    def objective(self, u, state, goal_state):
        """Compute objective function for optimization."""
        total_cost = 0
        current_state = state.copy()
        
        for i in range(self.horizon):
            # Reshape control input
            control = u[2*i:2*(i+1)]
            
            # State error
            state_error = current_state - goal_state
            
            # Compute costs
            state_cost = state_error.T @ self.Q @ state_error
            control_cost = control.T @ self.R @ control
            total_cost += float(state_cost + control_cost)
            
            # Predict next state
            current_state = self.predict_state(current_state, control, self.dt)
        
        # Terminal cost
        final_error = current_state - goal_state
        terminal_cost = final_error.T @ self.P @ final_error
        total_cost += float(terminal_cost)
        
        return total_cost
    
    def get_control(self, current_state, goal_state, obstacles=None):
        """Compute optimal control sequence using MPC."""
        # Initial guess for control sequence
        u0 = np.zeros(2 * self.horizon)
        
        # Control constraints
        bounds = []
        for _ in range(self.horizon):
            bounds += [
                (-self.max_velocity, self.max_velocity),
                (-self.max_angular_velocity, self.max_angular_velocity)
            ]
            
        # Solve optimization problem
        result = minimize(
            self.objective,
            u0,
            args=(current_state, goal_state),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        if not result.success:
            print("Warning: MPC optimization failed")
            
        # Return first control action
        return result.x[:2]
    
    def check_safety(self, state, obstacles):
        """Check if state is safe (collision-free)."""
        if obstacles is None:
            return True
            
        robot_radius = 0.3  # Should match environment
        
        for obstacle in obstacles:
            dist = np.linalg.norm(state[:2] - obstacle[:2])
            if dist < (robot_radius + obstacle[2]):  # obstacle[2] is radius
                return False
                
        return True
    
    def get_safe_control(self, current_state, goal_state, obstacles=None):
        """Get control input with safety guarantees."""
        # Try MPC control first
        control = self.get_control(current_state, goal_state, obstacles)
        
        # Predict next state
        next_state = self.predict_state(current_state, control, self.dt)
        
        # If unsafe, use backup control law
        if not self.check_safety(next_state, obstacles):
            # Simple backup: stop and turn towards goal
            angle_to_goal = np.arctan2(
                goal_state[1] - current_state[1],
                goal_state[0] - current_state[0]
            )
            
            angle_diff = angle_to_goal - current_state[2]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            control = np.array([0.0, np.clip(
                angle_diff,
                -self.max_angular_velocity,
                self.max_angular_velocity
            )])
            
        return control 