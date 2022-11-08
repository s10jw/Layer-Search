import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import sys
import scipy.signal as signal
import time
from numba import jit
from shutil import copyfile
import os

class GridShift:
    thresh = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1], [-1, 1, -1], [-1, 1, 0],
              [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1], [0, 1, 0],
              [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0],
              [1, 1, 1]]

    def __init__(self, img, h):
        """
        Initializes GridShift instance, takes as input one BGR image as img, and a value h used to partition img into grid
        cells of side length h. Initializes empty hash table S (store centroid), C (store grid pixel count), and H
        (store resident data points).
        :param img:
        :param h:
        """
        self.h = h
        self.img = img
        self.height, self.width, self.channels = img.shape
        self.S = {}
        self.C = {}
        self.H = {}
        self.S_temp = self.S
        self.C_temp = self.C
        self.H_temp = self.H
        assert self.height % self.h == 0 and self.width % self.h == 0, 'Grid Cell size does not divide image.'

    def getNeighbors(self, keys):
        neighbors = {}
        count, mulsum = None, None
        for index in keys:
            for i in GridShift.thresh:
                val = np.add(index, i)
                if val in keys:
                    if index in neighbors.keys():
                        mulsum += self.C_temp[val] * self.S_temp[val]
                        count += self.C_temp
                        neighbors[index][0].append(val)
                        neighbors[index][1] += mulsum
                        neighbors[index][2] += count
                    else:
                        mulsum = self.C_temp[val] * self.S_temp[val]
                        count = self.C_temp[val]
                        neighbors[index] = [[val], mulsum, count]
        return neighbors

    def formGrids(self):
        raw_data = self.img.flatten()
        for i in range(len(raw_data)):
            index = np.floor(np.divide(raw_data[i], self.h))
            if index in self.S.keys():
                curr_sum = np.multiply(self.S[index], self.C[index])
                new_sum = np.add(curr_sum, raw_data[i])
                new_count = self.C[index] + 1

                self.S[index] = np.divide(new_sum, new_count)
                self.C[index] = new_count
                self.H[index].append(i)
            else:
                self.S[index] = raw_data[i]
                self.C[index] = 1
                self.H[index] = [i]

    def mergeGrids(self):
        # Find initial neighbor info
        neighbors = self.getNeighbors(self.S.keys())
        # Create a temporary copy of hash contents
        self.S_temp = self.S
        self.C_temp = self.C
        self.H_temp = self.H
        # Empty parent hash tables
        self.S = {}
        self.C = {}
        while neighbors.keys():
            # Create a temporary copy of hash contents
            S_temp = self.S
            C_temp = self.C
            H_temp = self.H
            # Empty parent hash tables
            self.S = {}
            self.C = {}
            for i in S_temp.keys():
                neighbor_sum = neighbors[i][1] + S_temp[i] * C_temp[i]
                neighbor_count = neighbors[i][2] + C_temp[i]
                S_temp[i] = np.divide(neighbor_sum, neighbor_count)
                index = np.floor(np.divide(S_temp[i], self.h))
                if S_temp[index] in self.S.keys():
                    curr_sum = np.multiply(self.S[index], self.C[index])
                    new_sum = np.add(curr_sum, S_temp[i])
                    new_count = self.C[index] + C_temp[i]

                    self.S[index] = np.divide(new_sum, new_count)
                    self.C[index] = new_count
                    self.H[index].append(i)
                else:
                    self.S[index] = S_temp[i]
                    self.C[index] = C_temp[i]
                    self.H[index] = H_temp[i]
                neighbors = self.getNeighbors(self.S.keys())

