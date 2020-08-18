import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread("bridge_shadow.jpg")
image = np.copy(img)

img2 = image[:, :, :]
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
thresh_min = 20
thresh_max = 100
sobel_th = [20, 100]
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= sobel_th[0]) & (scaled_sobel <= sobel_th[1])] = 1
histogram = np.sum(sxbinary, axis=0)
print(histogram.shape[0]/2)
# plt.plot(histogram)

# HLS conversion
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:, :, 0]
L = hls[:, :, 1]
S = hls[:, :, 2]
thresh = [170, 255]
binary = np.zeros_like(S)
binary[(S >= thresh[0]) & (S <= thresh[1])] = 1

combined = np.zeros_like(binary)
combined[(binary == 1) | (sxbinary == 1)] = 1

plt.imshow(combined, cmap='gray')
plt.show()
