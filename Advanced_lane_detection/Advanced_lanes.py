import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
# %matplotlib inline

# Functions included.
# Camera distortion function
# sobel and hsv combined binary image
# perspective transform function
# find the line equations and on image
# reverse perspective transform

# 1) Camera Distortion

image = glob.glob('../SelfDriving_Lesson2/Calibration images/GOPR00*.jpg')
objpoints = []
imgpoints = []

for frame in image:
    img = mpimg.imread(frame)
    objp = np.zeros((8*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

# filename = "project_video.mp4"
# video = cv2.VideoCapture(filename)
# while True:

# ret, frame = video.read()  # converts the video feed into image feed

def calibrate(input_image):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, input_image.shape[1::-1], None, None)
    dst = cv2.undistort(input_image, mtx, dist, None, mtx)
    return dst

# 2) Sobel and HSV operator:


def sobel_hsv_binary(input_image):
    s_limits = [20, 100]
    hls_limits = [170, 255]
    gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= s_limits[0]) & (scaled_sobel <= s_limits[1])] = 1

    hls = cv2.cvtColor(input_image, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    binary = np.zeros_like(s)
    binary[(s >= hls_limits[0]) & (s <= hls_limits[1])] = 1

    combined = np.zeros_like(binary)
    combined[(binary == 1) | (sxbinary == 1)] = 1
    return combined

def wrap(img):

    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[335, 660],
         [1086, 660],
         [570, 490],
         [777, 485]])

    dst = np.float32(
        [[335, 660],
         [1010, 660],
         [336, 61],
         [1010, 65]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def inv_wrap(img):

    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[335, 660],
         [1086, 660],
         [570, 490],
         [777, 485]])

    dst = np.float32(
        [[335, 660],
         [1010, 660],
         [336, 61],
         [1010, 65]])

    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M_inv, img_size, flags=cv2.INTER_LINEAR)
    return warped



def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50

    window_height = np.int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(binary_warped):

    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return left_fitx, right_fitx, ploty


def image_plot(binary_warped):
    left_fitx, right_fitx, ploty = fit_polynomial(binary_warped)
    blank_image = np.copy(binary_warped)*0
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return blank_image

# filename = "project_video.mp4"
# video = cv2.VideoCapture(filename)  # captures video feed

# while True:
#     ret, frame = video.read()  # converts the video feed into image feed
#     if not ret:
#         video = cv2.VideoCapture(filename)  # makes the video run in a continuous loop
#         continue
test_image = mpimg.imread('test6.jpg')
calibrate_image = calibrate(test_image)
binary = sobel_hsv_binary(calibrate_image)
binary_warped = wrap(binary)
plot = image_plot(binary_warped)
out_img_1 = np.dstack((plot, plot, plot)) * 255
out_img_2 = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
# combined = cv2.addWeighted()
plt.imshow(out_img_2)
plt.show()
# cv2.imshow('combined', out_img_2)
# cv2.imshow('video', frame)
# key = cv2.waitKey(1)
#     if key == 27:
#         break
# video.release()
# cv2.destroyAllWindows()

