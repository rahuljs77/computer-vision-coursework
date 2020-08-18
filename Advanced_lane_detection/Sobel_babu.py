import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

"""
    The sobel operator differentiates a 3X3 area in the image and checks for change in intensity, the resultant matrix 
    is the then converted to absolute to get rid of the negative values. This matrix is then converted into 8 bit image
    by multiplying it with 255 accordingly. Using thresholds they are converted into binary images 
"""

image = mpimg.imread("bridge_shadow.jpg")
plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

abs_sobelx = np.absolute(sobelx)
abs_sobely = np.absolute(sobely)

gradmag = np.sqrt(sobelx**2 + sobely**2)

scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
scaled_sobel2 = np.uint8(255*gradmag/np.max(gradmag))

thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary2 = np.zeros_like(scaled_sobel2)


sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
sxbinary2[(scaled_sobel2 >= thresh_min) & (scaled_sobel2 <= thresh_max)] = 1

plt.imshow(sxbinary2, cmap='gray')
plt.show()
