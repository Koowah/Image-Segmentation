import basictools as bt
import numpy as np

path = './images_BW/alfa2.bmp'

image, shape = bt.read_image(path)

print(type(image))
print(np.unique(image))

bt.display_image('Hello', image)