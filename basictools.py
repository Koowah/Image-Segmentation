from sklearn.cluster import KMeans
import numpy as np
import cv2

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # this function allows to always display images with respect to ratio - needed for concatenation of images in particular
    # credit goes to nathancy from StackOverflow
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def read_image(path : str):
    image = cv2.imread(path)
    image //= 255 # anticipating use of gaussian noise 
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # making sure it's greyscale
    return grey_img, np.shape(grey_img)

def display_image(title:str, image):
    cv2.imshow(title + ' (press any key to close window)', ResizeWithAspectRatio(image, width=640)) # cv2.resize(image, (1024,576)) instead of image to specify size depending on device
    cv2.waitKey()
    cv2.destroyAllWindows()

def identify_classes(X):
    return np.unique(X)

def gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2):
    # adding gaussian noise to each class separatly
    gaussian_class_1 = np.random.normal(m1, sig1, (m,n))
    gaussian_class_1[X == cl2] = 0
    gaussian_class_2 = np.random.normal(m2, sig2, (m,n))
    gaussian_class_2[X == cl1] = 0
    noise = gaussian_class_1 + gaussian_class_2

    return noise

def error_rate(tolerance, A, B, m, n):
    return 1 - (abs(A - B) <= tolerance).sum()/(m*n) # rate of different pixels between A & B

def evaluate_kmeans(k, A, B):
    shape = A.shape
    
    model = KMeans(n_clusters=k).fit(B.reshape(-1,1)) 
    # .clip(0, x) empirically gives better results in our specific case with x>1
    # the intuition is that here since we've got a measure of proximity
    # there are more chances that occurences above 1 stem from a gaussian noise added to a point where class is 1
    # so we stack occurences bellow 0 at 0, and let those above 1 draw the centroids towards 1
    
    values = model.cluster_centers_
    labels = model.labels_
    img_seg = values[labels].reshape(shape) # KM segmented image
    
    # adapting classes to our problem
    unique_values = np.unique(img_seg)
    img_seg[img_seg == unique_values[0]] = 0
    img_seg[img_seg == unique_values[1]] = 1
    
    return img_seg, error_rate(0, img_seg, A, shape[0], shape[1])