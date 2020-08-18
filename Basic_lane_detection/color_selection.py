import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('realmadrid.jpg')

print(image.shape)

ysize = image.shape[0]
xsize = image.shape[1]
image2 = np.copy(image)
image3 = np.copy(image)

red = green = blue = 170
threshold = [red, green, blue]
plt.imshow(image)

limits = (image2[:, :, 0] < threshold[0]) | (image2[:, :, 1] < threshold[1]) | (image2[:, :, 2] < threshold[2])
image2[limits] = [0, 0, 0]

left_top = [0, 0]
right_top = [1920, 0]
apex = [1920/2, 1907]

fit_left = np.polyfit((left_top[0], apex[0]), (left_top[1], apex[1]), 1)
fit_right = np.polyfit((right_top[0], apex[0]), (right_top[1], apex[1]), 1)
fit_top = np.polyfit((left_top[0], right_top[0]), (left_top[1], right_top[1]), 1)

XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

region_masks = ((YY < (XX*fit_left[0] + fit_left[1])) & (YY < (XX*fit_right[0] + fit_right[1])) & (YY > (XX*fit_top[0] + fit_top[1])))

image2[region_masks] = [255, 0, 0]

image3[~limits & region_masks] = [255, 0, 0]

plt.imshow(image3)

plt.show()
