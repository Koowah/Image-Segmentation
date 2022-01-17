import numpy as np
import cv2

def read_image(path : str):
    image = cv2.imread(path)
    image //= 255 # anticipating use of gaussian noise 
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_img, np.shape(grey_img)

def display_image(title:str, image):
    cv2.imshow(title, cv2.resize(image, (1280,720))) # specify size depending on device
    cv2.waitKey()
    cv2.destroyAllWindows()

def identify_classes(X):
    return np.unique(X)

def bruit_gauss(X, m, n, cl1, cl2, m1, sig1, m2, sig2):
    gaussian_class_1 = np.random.normal(m1, sig1, (m,n))
    gaussian_class_1[X == cl2] = 0
    gaussian_class_2 = np.random.normal(m2, sig2, (m,n)) # will probably have to use np.where class == {1,2}
    gaussian_class_2[X == cl1] = 0
    noise = gaussian_class_1 + gaussian_class_2

    return noise