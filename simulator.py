import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



class Robot:
    def __init__(self):
        self.episode_length = 100
        self.links = []
        self.goal = self.get_random_goal()

        self.DH_notations = np.array([[0, -np.pi/2, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 1],
                            [0, 0, 0, 1]])

        self.update_robot_shape()
        self.tick = 0


    def create_link(self, DH_notation):
        theta, alpha, d, r = DH_notation
        A_n = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), r*np.cos(theta)],
                        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), r*np.sin(theta)],
                        [0, np.sin(alpha), np.cos(alpha), d],
                        [0, 0, 0, 1]])
        return A_n

    def update_robot_shape(self):
        self.links = []
        for DH_notation in self.DH_notations:
            self.links.append(self.create_link(DH_notation))

    def move_to(self, angles):
        angles = np.array(angles).reshape((1, 4))
        self.DH_notations = self.DH_notations[:, -3:]
        self.DH_notations = np.concatenate((angles.T, self.DH_notations), axis=1)
        self.update_robot_shape()

    def take_action(self, angle_differences):
        angle_differences = np.array(angle_differences).reshape((1, 4))
        current_angles = self.DH_notations[:, 0]
        new_angles = current_angles + angle_differences
        self.DH_notations = self.DH_notations[:, -3:]
        self.DH_notations = np.concatenate((new_angles.T, self.DH_notations), axis=1)
        self.update_robot_shape()

    def get_robot_position(self):
        link1 = self.links[0]
        link2 = np.matmul(link1, self.links[1])
        link3 = np.matmul(link2, self.links[2])
        link4 = np.matmul(link3, self.links[3])
        link1 = self.extract_link_position(link1)
        link2 = self.extract_link_position(link2)
        link3 = self.extract_link_position(link3)
        link4 = self.extract_link_position(link4)
        return np.array([link1, link2, link3, link4])

    def extract_link_position(self, link):
        position = {}
        x = link[0, 3]
        y = link[1, 3]
        z = link[2, 3]
        position['xyz'] = [x, y, z]
        x_angle = np.arctan2(link[2, 1], link[2, 2])
        y_angle = np.arctan2(-link[2, 0], np.sqrt(link[2, 1]**2 + link[2, 2]**2))
        z_angle = np.arctan2(link[1, 0], link[0, 0])
        position['angle'] = [x_angle, y_angle, z_angle]
        return position

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        joint_positions = self.get_robot_position()
        stacked_joints = [joint['xyz'] for joint in joint_positions]
        stacked_joints = np.concatenate(([[0, 0, 0]], stacked_joints))
        ax.plot(stacked_joints[:, 0], stacked_joints[:, 1], stacked_joints[:, 2])
        plt.show()

    def get_random_goal(self):
        x, y, z = np.random.uniform(high=4, size=3)
        return np.array([x, y, z])

    def get_reward(self):
        x, y, z = self.get_robot_position()[-1]['xyz']
        xg, yg, zg = self.goal
        reward = -np.sqrt((x-xg)**2 + (y-yg)**2 + (z-zg)**2)
        return reward

    def flatten_state(self, joints):
        state = np.array([[joint['xyz'], joint['angle']] for joint in joints])
        state = np.append(state, self.goal)
        state = state.flatten()
        return state

    def reset(self):
        self.tick = 0
        self.goal = self.get_random_goal()
        random_angles = np.random.uniform(0, np.pi*2, 4)
        self.move_to(random_angles)
        state = self.get_robot_position()
        return self.flatten_state(state)

    def step(self, action):
        self.take_action(action)
        state = self.flatten_state(self.get_robot_position())
        reward = self.get_reward()
        done = self.tick > self.episode_length
        self.tick += 1
        return [state, reward, done]

robot = Robot()
print(robot.get_reward())
robot.plot()
robot.step([0.1, 0.1, 0.1, 0.1])
print(robot.get_reward())

robot.step([0.1, 0.1, 0.1, 0.1])
print(robot.get_reward())

robot.step([0.1, 0.1, 0.1, 0.1])
print(robot.get_reward())

robot.step([0.1, 0.1, 0.1, 0.1])
print(robot.get_reward())
robot.plot()


