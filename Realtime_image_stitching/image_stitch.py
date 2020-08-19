from panorama import Panaroma
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

no_of_images = 2
left_image = mpimg.imread("left_cam.jpg")
right_image = mpimg.imread("right_cam.jpg")
right_image = right_image[0:801, 0:1267]
print(left_image.shape)
print(right_image.shape)
images = [left_image, right_image]
panaroma = Panaroma()
(result, matched_points) = panaroma.image_stitch([images[0], images[1]], match_status=True)
plt.imshow(result)

plt.show()
