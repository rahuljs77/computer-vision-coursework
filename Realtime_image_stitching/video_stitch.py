from panorama import Panaroma
import cv2
import numpy as np

video_left = cv2.VideoCapture('video_left.mp4')
video_right = cv2.VideoCapture('video_right.mp4')
video = cv2.VideoCapture('cr7.mp4')

while True:
    rets, frame = video.read()
    ret, frame_left = video_left.read()
    ret2, frame_right = video_right.read()
    no_of_images = 2
    images = [frame_left, frame_right]

    panaroma = Panaroma()

    (result, matched_points) = panaroma.image_stitch([images[0], images[1]], match_status=True)
    final = result[0:360, 0:630]

    cv2.imshow('result', final)
    cv2.imshow('left', frame_left)
    cv2.imshow('right', frame_right)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()


