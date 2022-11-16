import cv2
import numpy as np

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
        self.raw_data = img.reshape(self.height * self.width, self.channels)
        self.S = {}
        self.C = {}
        self.H = {}
        self.S_temp = self.S
        self.C_temp = self.C
        self.H_temp = self.H
        assert self.height % self.h == 0 and self.width % self.h == 0, 'Grid Cell size does not divide image.'

    def formGrids(self):
        for i in range(len(self.raw_data)):
            index = np.floor(np.divide(self.raw_data[i], self.h)).tobytes()
            if index in self.S.keys():
                curr_sum = np.multiply(self.S[index], self.C[index])
                new_sum = np.add(curr_sum, self.raw_data[i])
                new_count = self.C[index] + 1

                self.S[index] = np.divide(new_sum, new_count)
                self.C[index] = new_count
                self.H[index].append(i)
            else:
                self.S[index] = self.raw_data[i]
                self.C[index] = 1
                self.H[index] = [i]

    def mergeGrids(self):
        neighbors = {}
        check = True
        # Create a temporary copy of hash contents
        self.S_temp = self.S
        self.C_temp = self.C
        self.H_temp = self.H
        # Empty parent hash tables
        self.S = {}
        self.C = {}

        while neighbors.keys() or check:
            if check:
                check = False
            # Create a temporary copy of hash contents
            S_temp = self.S
            C_temp = self.C
            H_temp = self.H
            # Empty parent hash tables
            self.S = {}
            self.C = {}
            for i in S_temp.keys():
                # First, we find all neighboring active grids to the ith index
                neighbors = {}
                count, mulsum = None, None
                for j in GridShift.thresh:
                    val = np.add(i, j)
                    if val in S_temp.keys():
                        if i in neighbors.keys():
                            mulsum += self.C_temp[val] * self.S_temp[val]
                            count += self.C_temp
                            neighbors[i][0].append(val)
                            neighbors[i][1] += mulsum
                            neighbors[i][2] += count
                        else:
                            mulsum = self.C_temp[val] * self.S_temp[val]
                            count = self.C_temp[val]
                            neighbors[i] = [[val], mulsum, count]

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

    def segment(self):
        self.formGrids()
        self.mergeGrids()

        segmented_data = self.raw_data

        for pair in self.H.items():
            segmented_data[pair[1]] = pair[0]

        return segmented_data

img = cv2.imread('Assets/pepper_test.png')
# cv2.imshow('test', img)


radius = 5
segmented = GridShift(img, radius)
segmented.segment()
print('done')

cv2.waitKey(0)
cv2.destroyAllWindows()

