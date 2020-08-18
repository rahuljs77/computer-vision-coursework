import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

image = mpimg.imread('solidYellowCurve.jpg')
image2 = np.copy(image)

print(image.shape)

# check for image shape to be in X x Y coordinates

#  converting the image into greyscale using cv2
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# softening the image using gaussian blue
kernel = 9
blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

# edge detection using canny algorithm
low = 20
high = 150
edges = cv2.Canny(blur, low, high)

# line detection using hough transform

rho = 2
theta = np.pi / 180
threshold = 10
min_line_length = 5
max_line_gap = 20
line_image = np.copy(image) * 0  # blank image

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)  # gives the coordinates of lines start to finish

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
for line in lines:
    for x1, y1, x2, y2 in line:
        if (region_masks[y1][x1]) and (region_masks[y2][x2]):  # checking if they are inside the masked region
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 20), 5)


# converting binary image to color image
color_edges = np.dstack((edges, edges, edges))

# overlaying one image on top of another
combined = cv2.addWeighted(image2, 0.8, line_image, 1, 0)
# plt.imshow(combined)
# plt.imshow(edges, cmap="Greys_r")
# plt.imshow(gray, cmap='gray')
plt.imshow(line_image)
plt.show()
