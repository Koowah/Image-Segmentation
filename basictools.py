import numpy as np
import cv2 as cv

def read_image(path : str):
    image = cv.imread(path)
    image //= 255
    grey_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return grey_img, np.shape(grey_img)

def display_image(title:str, image):
    cv.imshow(title, image)
    cv.waitKey()

def identif_classes(X):
    return np.unique(X)

def bruit_gauss(X, m, n, cl1, cl2, m1, sig1, m2, sig2):
    gaussian_class_1 = np.random.normal(m1, sig1, (m,n))
    gaussian_class_1[X == cl2] = 0
    gaussian_class_2 = np.random.normal(m2, sig2, (m,n))
    gaussian_class_2[X == cl1] = 0
    noisy_X = X + gaussian_class_1 + gaussian_class_2

    return noisy_X