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
        assert self.height % self.h == 0 and self.width % self.h == 0, 'Grid Cell size does not divide image.'

    def formGrids(self):
        raw_data = self.img.flatten()
        y = 1
        index = -1
        hold = self.width / self.h
        row = 0
        for i in range(len(raw_data)):
            x = i + 1
            # Creates index for an active grid of size h x h, commits all pixels with same index to the same dict.

            if x % self.h + 1 == 0:
                if x % self.width + 1 == 0:
                    y += 1
                    if y % self.h + 1 == 0:
                        row += 1
                        index = row * hold
                    else:
                        index = row * hold
                else:
                    index += 1

            if index in self.S.keys():
                curr_sum = np.multiply(self.S[index], self.C[index])
                new_sum = np.add(curr_sum, raw_data[i])
                new_count = self.C[index] + 1

                self.S[index] = np.divide(new_sum, new_count)
                self.C[index] = new_count
                self.H[index].add(i)
            else:
                self.S[index] = raw_data[i]
                self.C[index] = 1
                self.H[index] = {i}

    def mergeGrids(self):
        while self.S.keys():
            # Create a temporary copy of hash contents
            S_temp = self.S
            C_temp = self.C
            H_temp = self.H
            # Empty parent hash tables
            self.S = {}
            self.C = {}
            for i in S_temp.keys():
                neighbor_sum = 0
                neighbor_count = 0
                for j in range(-1, 2):
                    neighbor_sum += np.multiply(C_temp[i + j], S_temp[i + j])
                    neighbor_count += C_temp[i + j]
                S_temp[i] = np.divide(neighbor_sum, neighbor_count)
                index = np.floor(np.divide(S_temp[i], self.h))
                if S_temp[index] in self.S.keys():
                    curr_sum = np.multiply(self.S[index], self.C[index])
                    new_sum = np.add(curr_sum, S_temp[i])
                    new_count = self.C[index] + C_temp[i]

                    self.S[index] = np.divide(new_sum, new_count)
                    self.C[index] = new_count
                    self.H[index].add(i)
                else:
                    self.S[index] = S_temp[i]
                    self.C[index] = C_temp[i]
                    self.H[index] = H_temp[i]

