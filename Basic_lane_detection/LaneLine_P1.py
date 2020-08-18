import cv2
import numpy as np

filename = "solidWhiteRight.mp4"
video = cv2.VideoCapture(filename)  # captures video feed

while True:
    ret, frame = video.read()  # converts the video feed into image feed
    if not ret:
        video = cv2.VideoCapture(filename)  # makes the video run in a continuous loop
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # softening the image using gaussian blue
    kernel = 9
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    # edge detection using canny algorithm
    low = 50
    high = 150
    edges = cv2.Canny(blur, low, high)

    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 10  
    max_line_gap = 1
    line_image = np.copy(frame) * 0  # blank image

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)  # gives the coordinates of lines start to finish

    # region masking by making a triangle shape on the image
    left = [0, 540]
    right = [960, 540]
    apex = [480, 310]

    # A and B coefficients of a line
    left_limits = np.polyfit((left[0], apex[0]), (left[1], apex[1]), 1)
    right_limits = np.polyfit((right[0], apex[0]), (right[1], apex[1]), 1)
    bottom_limits = np.polyfit((left[0], right[0]), (left[1], right[1]), 1)

    # position of each pixel(xx and yy)
    XX, YY = np.meshgrid(np.arange(0, frame.shape[1]), np.arange(0, frame.shape[0]))
    region_masks = ((YY > (XX * left_limits[0] + left_limits[1])) & (YY > (XX * right_limits[0] + right_limits[1])) & (
            YY < (XX * bottom_limits[0] + bottom_limits[1])))

    # Drawing those line a black image
    m_positive = []
    m_negative = []
    b_left = []
    b_right = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if (region_masks[y1][x1]) and (region_masks[y2][x2]):  # checking if they are inside the masked region
                if x2 == x1:
                    pass
                else:
                    m = (y2 - y1) / (x2 - x1)
                    if m == (float('inf')):
                        pass
                    else:
                        b = y2 - m * x2
                        if 0 > m > -2:  # it is a right lane
                            m_negative.append(m)
                            b_right.append(b)
                        elif 2 > m > 0:  # it is a left lane
                            m_positive.append(m)
                            b_left.append(b)
    m_right = np.mean(m_negative)
    b_right = np.mean(b_right)

    Yr_start = frame.shape[0]
    Xr_start = int((Yr_start - b_right) / m_right)

    Yr_end = 350
    Xr_end = int((Yr_end - b_right) / m_right)
    cv2.line(line_image, (Xr_start, Yr_start), (Xr_end, Yr_end), (0, 255, 0), 10)

    m_left = np.mean(m_positive)
    b_left = np.mean(b_left)

    Yl_start = frame.shape[0]
    Xl_start = int((Yl_start - b_left) / m_left)

    Yl_end = 350
    Xl_end = int((Yl_end - b_left) / m_left)
    cv2.line(line_image, (Xl_start, Yl_start), (Xl_end, Yl_end), (0, 255, 0), 10)

    # converting binary image to color image
    color_edges = np.dstack((edges, edges, edges))

    # overlaying one image on top of another
    combined_lanes = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    cv2.imshow('combined', combined_lanes)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
