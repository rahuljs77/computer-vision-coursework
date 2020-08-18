import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


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
        [[200, 670],
         [1200, 670],
         [560, 460],
         [745, 460]])

    dst = np.float32(
        [[300, 660],
         [1020, 660],
         [300, 60],
         [1020, 60]])

    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M_inv, img_size, flags=cv2.INTER_LINEAR)
    return warped

img = mpimg.imread("test6.jpg")
warped = wrap(img)
warped_inv = inv_wrap(warped)
plt.imshow(img)
plt.show()
plt.imshow(warped)
plt.show()
plt.imshow(warped_inv)
plt.show()

