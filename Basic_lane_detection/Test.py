# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# image = mpimg.imread('realmadrid.jpg')
# bottom = image[image.shape[0]//2::, :]
# gray = cv2.cvtColor(bottom, cv2.COLOR_RGB2GRAY)
# limits = [50, 150]
# binary = np.zeros_like(gray)
# binary[(gray >= limits[0]) & (gray <= limits[1])] = 1
# histogram = np.sum(binary, axis=0)
# nonzeros = binary.nonzero()
# nonzeroy = np.array(nonzeros[0])
# nonzerox = nonzeros[1]
# y_limits = [400, 700]
# good_list = []
# good_indices = ((nonzeroy > y_limits[0]) & (nonzeroy < y_limits[1])).nonzero()[0]
# good_list.append(good_indices)
# good_indices2 = ((nonzerox > y_limits[0]) & (nonzerox < y_limits[1])).nonzero()[0]
# good_list.append(good_indices2)
# ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
# # good_list = np.concatenate(good_list)
# # print(plotx)
# # plt.plot(histogram)
#
# # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# # a1 = np.nonzero(a > 3)
# print(len(good_indices))
# print(len(nonzeroy))
# plt.imshow(binary, cmap='gray')
# plt.show()

import numpy as np

W = np.array(np.random(1, 2))
print(W)
