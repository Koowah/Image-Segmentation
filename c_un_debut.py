import basictools as bt
import numpy as np

path = './images_BW/alfa2.bmp'
print(f'Image : {path.split("/")[-1]}')

image, shape = bt.read_image(path)
cl1, cl2 = bt.identify_classes(image)
print(f'classe 1 : {cl1}, classe 2 : {cl2}')

bt.display_image('Image', 255* image)

noisy = image + bt.bruit_gauss(image, shape[0], shape[1],cl1, cl2, .5, 0, 0, 0)
noisy_1 = noisy + bt.bruit_gauss(image, shape[0], shape[1],cl1, cl2, 0, 0, 0, 1)
bt.display_image("Classe 1 bruit de moyenne .5", noisy)
bt.display_image('+ Classe 2 bruit d\'écart type 1', noisy_1)