import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML


image = mpimg.imread('solidYellowLeft.jpg')
# image = mpimg.imread('solidYellowCurve.jpg')
# image = mpimg.imread('solidYellowCurve2.jpg')
# image = mpimg.imread('solidWhiteRight.jpg')
# image = mpimg.imread('solidWhiteCurve.jpg')
# image = mpimg.imread('whiteCarLaneSwitch.jpg')



image2 = np.copy(image)

#  converting the image into greyscale using cv2
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray, cmap='gray')

# softening the image using gaussian blue
kernel = 9
blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

# edge detection using canny algorithm
low = 50
high = 150
edges = cv2.Canny(blur, low, high)

# plt.imshow(edges,  cmap="Greys_r")
# line detection using hough transform

rho = 2
theta = np.pi / 180
threshold = 22  # 15
min_line_length = 44
max_line_gap = 25
line_image = np.copy(image) * 0  # blank image

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap) # gives the coordinates of lines start to finish

# region masking
left = [0, 540]
right = [960, 540]
apex = [480, 310]

# A and B coefficients of a line
left_limits = np.polyfit((left[0], apex[0]), (left[1], apex[1]), 1)
right_limits = np.polyfit((right[0], apex[0]), (right[1], apex[1]), 1)
bottom_limits = np.polyfit((left[0], right[0]), (left[1], right[1]), 1)

# position of each pixel(xx and yy)
XX, YY = np.meshgrid(np.arange(0, image.shape[1]), np.arange(0, image.shape[0]))
region_masks = ((YY > (XX * left_limits[0] + left_limits[1])) & (YY > (XX * right_limits[0] + right_limits[1])) & (
            YY < (XX * bottom_limits[0] + bottom_limits[1])))

# Drawing those line a black image
X_left = []
X_right = []
Y_left = []
Y_right = []
m_positive = []
m_negative = []
b_left = []
b_right = []

for line in lines:
    for x1, y1, x2, y2 in line:
        if (region_masks[y1][x1]) and (region_masks[y2][x2]):  # checking if they are inside the masked region
            # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 20), 5)
            if x2 == x1:
                pass
            else:
                m = (y2 - y1)/(x2 - x1)
                if m == (float('inf')):
                    pass
                else:
                    b = y2 - m*x2
                    if 0 > m > -1:  # it is a right lane
                        # X_right += [x1, x2]
                        # Y_right += [y1, y2]
                        m_negative.append(m)
                        b_right.append(b)
                    elif 1 > m > 0:  # it is a left lane
                        # print(m)
                        # X_left += [x1, x2]
                        # Y_left += [y1, y2]
                        m_positive.append(m)
                        b_left.append(b)

############################ Extrapolating ###########################################
# Z_right = np.polyfit(X_right, Y_right, 1)
# Yr_start = image.shape[0]
# Xr_start = int((Yr_start - Z_right[1])/Z_right[0])
#
# Yr_end = min(Y_right)
# Xr_end = int((Yr_end - Z_right[1])/Z_right[0])
#
# cv2.line(line_image, (Xr_start, Yr_start), (Xr_end, Yr_end), (255, 0, 20), 7)
#
# Z_left = np.polyfit(X_left, Y_left, 1)
# Yl_start = image.shape[0]
# print(Yl_start)
# Xl_start = int((Yl_start - Z_left[1])/Z_left[0])
#
# Yl_end = min(Y_left)
# Xl_end = int((Yl_end - Z_left[1])/Z_left[0])
#
# cv2.line(line_image, (Xl_start, Yl_start), (Xl_end, Yl_end), (255, 0, 20), 7)
#################################################################################

m_right = np.mean(m_negative)
b_right = np.mean(b_right)

Yr_start = image.shape[0]
Xr_start = int((Yr_start - b_right)/m_right)

Yr_end = 350
Xr_end = int((Yr_end - b_right)/m_right)
cv2.line(line_image, (Xr_start, Yr_start), (Xr_end, Yr_end), (255, 0, 20), 10)

m_left = np.mean(m_positive)
b_left = np.mean(b_left)

Yl_start = image.shape[0]
Xl_start = int((Yl_start - b_left)/m_left)

Yl_end = 350
Xl_end = int((Yl_end - b_left)/m_left)
cv2.line(line_image, (Xl_start, Yl_start), (Xl_end, Yl_end), (255, 0, 20), 10)
##########################################################################################

# converting binary image to color image
color_edges = np.dstack((edges, edges, edges))

# overlaying one image on top of another
combined = cv2.addWeighted(image2, 0.8, line_image, 1, 0)
plt.imshow(combined)
cv2.imwrite('combined.png', combined)
# plt.imshow(edges, cmap="Greys_r")
plt.show()
