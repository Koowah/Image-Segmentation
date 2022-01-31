from sklearn.cluster import KMeans
import numpy as np
import cv2

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_img, np.shape(grey_img)

def display_image(title:str, image):
    cv2.imshow(title + ' (press any key to close window)', ResizeWithAspectRatio(image, width=640)) # cv2.resize(image, (1024,576)) instead of image to specify size depending on device
    cv2.waitKey()
    cv2.destroyAllWindows()

def identify_classes(X):
    return np.unique(X)

def gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2):
    gaussian_class_1 = np.random.normal(m1, sig1, (m,n))
    gaussian_class_1[X == cl2] = 0
    gaussian_class_2 = np.random.normal(m2, sig2, (m,n)) # will probably have to use np.where class == {1,2}
    gaussian_class_2[X == cl1] = 0
    noise = gaussian_class_1 + gaussian_class_2

    return noise

def error_rate(tolerance, A, B, m, n):
    return 1 - ((abs(A.clip(0,1) - B.clip(0,1)) <= tolerance).sum())/(m*n)

def evaluate_kmeans(k, A, B):
    shape = A.shape
    
    model = KMeans(n_clusters=k, random_state=0).fit(B.clip(0,1).reshape(-1,1))
    values = model.cluster_centers_
    labels = model.labels_
    img_seg = values[labels].reshape(shape)
    
    return img_seg, error_rate(0.1, img_seg, A, shape[0], shape[1])