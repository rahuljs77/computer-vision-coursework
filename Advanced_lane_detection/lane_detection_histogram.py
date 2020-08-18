import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('realmadrid.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
bottom_half = gray[gray.shape[0]//2::, :]
thresh = [50, 100]
binary = np.zeros_like(bottom_half)
binary[(bottom_half > thresh[0]) & (bottom_half < thresh[1])] = 1

histogram = np.sum(binary, axis=0)
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[0:midpoint])

nonzero = binary.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

print(nonzeroy)
print(binary.shape)

# plt.plot(histogram)
plt.imshow(binary, cmap='gray')
plt.show()
