#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from drl_agent import DRLAgent
from mpc_controller import MPCController

class WarehouseNavigationNode:
    def __init__(self):
        rospy.init_node('warehouse_navigation_node')
        
        # Initialize controllers
        self.drl_agent = DRLAgent()
        self.mpc_controller = MPCController()
        
        # Load trained DRL model
        self.drl_agent.load("final_model")
        
        # Controller parameters
        self.use_drl = True  # Switch between DRL and MPC
        self.safety_threshold = 0.5  # Minimum distance for safety
        
        # Robot state
        self.current_pose = None
        self.current_goal = None
        self.laser_data = None
        
        # ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        
        # Timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Warehouse navigation node initialized")
        
    def odom_callback(self, msg):
        """Handle odometry data."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # Convert quaternion to euler angles
        _, _, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        
        self.current_pose = np.array([
            position.x,
            position.y,
            yaw
        ])
        
    def laser_callback(self, msg):
        """Handle laser scan data."""
        self.laser_data = np.array(msg.ranges)
        
    def goal_callback(self, msg):
        """Handle new navigation goal."""
        position = msg.pose.position
        orientation = msg.pose.orientation
        
        # Convert quaternion to euler angles
        _, _, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        
        self.current_goal = np.array([
            position.x,
            position.y,
            yaw
        ])
        
        rospy.loginfo(f"New goal received: {self.current_goal}")
        
    def get_obstacles_from_laser(self):
        """Convert laser scan to obstacle list."""
        if self.laser_data is None:
            return None
            
        obstacles = []
        angle_min = -np.pi/2
        angle_increment = np.pi / len(self.laser_data)
        
        for i, range_value in enumerate(self.laser_data):
            if range_value < self.safety_threshold:
                angle = angle_min + i * angle_increment
                x = range_value * np.cos(angle)
                y = range_value * np.sin(angle)
                obstacles.append([x, y, 0.1])  # 0.1m radius for point obstacles
                
        return obstacles if obstacles else None
        
    def check_safety(self):
        """Check if current state is safe."""
        if self.laser_data is None:
            return True
            
        return np.min(self.laser_data) > self.safety_threshold
        
    def control_loop(self, event):
        """Main control loop."""
        if None in (self.current_pose, self.current_goal, self.laser_data):
            return
            
        # Create observation for DRL agent
        observation = np.concatenate([
            self.current_pose,
            self.current_goal,
            self.laser_data
        ])
        
        # Get control command
        if self.use_drl and self.check_safety():
            # Use DRL when safe
            action, _ = self.drl_agent.model.predict(observation, deterministic=True)
            linear_vel, angular_vel = action
        else:
            # Fall back to MPC when unsafe or as backup
            obstacles = self.get_obstacles_from_laser()
            control = self.mpc_controller.get_safe_control(
                self.current_pose,
                self.current_goal,
                obstacles
            )
            linear_vel, angular_vel = control
            
        # Create and publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = float(linear_vel)
        cmd_vel.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(cmd_vel)
        
    def run(self):
        """Run the node."""
        rospy.spin()

if __name__ == '__main__':
    try:
        node = WarehouseNavigationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass 