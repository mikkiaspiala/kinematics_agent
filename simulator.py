import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



class Robot:
    def __init__(self, DH_notations):
        self.links = []
        self.update_robot_shape(DH_notations)


    def create_link(self, DH_notation):
        theta, alpha, d, r = DH_notation
        A_n = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), r*np.cos(theta)],
                        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), r*np.sin(theta)],
                        [0, np.sin(alpha), np.cos(alpha), d],
                        [0, 0, 0, 1]])
        return A_n

    def update_robot_shape(self, DH_notations):
        self.links = []
        for DH_notation in DH_notations:
            self.links.append(self.create_link(DH_notation))

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
        print(stacked_joints)
        ax.plot(stacked_joints[:, 0], stacked_joints[:, 1], stacked_joints[:, 2])
        plt.show()

DH_notations = [[0, -np.pi/2, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1]]

robot = Robot(DH_notations)
robot.plot()

DH_notations = [[np.pi, -np.pi/2, 1, 0],
                [0.12, 0, 0, 1],
                [0.12, 0, 0, 1],
                [0.12, 0, 0, 1]]

robot.update_robot_shape(DH_notations)
robot.plot()
