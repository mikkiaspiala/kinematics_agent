import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



class Robot:
    def __init__(self):
        self.links = []
        self.angles = np.array([0, 0, 0, 0])
        self.base = 1
        self.shoulder = 0.12
        self.bicep = 0.12
        self.forearm = 0.12

        self.base_length = 2
        self.shoulder_length = 1
        self.bicep_length = 1
        self.forearm_length = 1
        self.scale = 1

    def get_link_positions1(self):
        link0 = np.array([0, 1, 0]) * self.scale

        link1 = np.array([np.cos(self.base) * np.cos(self.shoulder) + link0[0], np.cos(self.base) * np.sin(self.shoulder) + link0[1],  np.sin(self.base) + link0[2]]) * self.scale

        link2 = np.array([np.cos(self.base) * np.cos(self.bicep) + link1[0], np.cos(self.base) * np.sin(self.bicep) + link1[1], np.sin(self.base) + link1[2]]) * self.shoulder_length * self.scale

        link3 = np.array([np.cos(self.base)*np.cos(self.bicep) + link2[0], np.cos(self.base)*np.sin(self.bicep) + link2[1], np.sin(self.base) + link2[2]]) * self.bicep_length * self.scale

        link4 = np.array([np.cos(self.base)*np.cos(self.forearm) + link3[0], np.cos(self.base)*np.sin(self.forearm) + link3[1], np.sin(self.base) + link3[2]]) * self.forearm_length * self.scale

        return np.array([[0, 0, 0], link0, link1, link2, link3, link4])

    def create_link(self, DH_notation):
        theta, alpha, d, r = DH_notation
        A_n = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), r*np.cos(theta)],
                        [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), r*np.sin(theta)],
                        [0, np.sin(alpha), np.cos(alpha), d],
                        [0, 0, 0, 1]])
        return A_n

    def define_robot_shape(self, DH_notations):
        for DH_notation in DH_notations:
            self.links.append(self.create_link(DH_notation))

    def get_robot_position(self):
        position = 1
        for link in self.links:
            pass

    def get_link_positions(self):
        link0 = np.array([0, 1, 0])

        link1 = np.array([(np.cos(self.shoulder) + link0[0])*np.cos(self.base), np.sin(self.shoulder) + link0[1], np.sin(self.base)])

        link2 = np.array([(np.cos(self.bicep + self.shoulder) + link1[0])*np.cos(self.base), np.sin(self.bicep + self.shoulder) + link1[1], (np.cos(self.bicep + self.shoulder) + link1[0])*np.sin(self.base)])

        link3 = np.array([(np.cos(self.bicep + self.forearm + self.shoulder) + link2[0])*np.cos(self.base), np.sin(self.forearm + self.bicep + self.shoulder) + link2[1], (np.cos(self.bicep + self.forearm + self.shoulder) + link2[0])*np.sin(self.base)])

        return np.array([[0, 0, 0], link0, link1, link2, link3])

    def plot_pose1(self):
        pose = self.get_link_positions()
        plt.plot(pose[:, 0], pose[:, 1])
        plt.show()

def plot_pose():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(0, np.pi*2)
    robot = Robot()
    for value in x:
        robot.base = value
        links = robot.get_link_positions()
        ax.plot(links[:, 0], links[:, 1], links[:, 2])
    plt.show()

plot_pose()
