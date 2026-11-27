# Data Detective: a photo is an array.
# A colour photo is a 3D array: rows, columns, and 3 numbers (red, green, blue) for every dot.
# Run this in Google Colab, one plt.imshow line at a time, to see each picture.
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

img = load_sample_image("china.jpg")
print(type(img))
print(img.shape, img.dtype)       # (427, 640, 3) uint8: whole numbers from 0 to 255
print(img[0, 0])                  # the top-left dot: [174 201 231]

plt.imshow(img)                        # the photo
# plt.imshow(img[100:300, 200:500])    # a crop: rows 100 to 299, columns 200 to 499
# plt.imshow(img[::-1])                # upside down: the rows are reversed
# plt.imshow(img[:, ::-1])             # a mirror image: the columns are reversed
# plt.imshow(img[::4, ::4])            # a smaller photo, like a thumbnail on a restaurant card
# plt.imshow(255 - img)                # the negative
