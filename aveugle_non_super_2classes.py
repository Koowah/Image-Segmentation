from basictools import *
from scipy.stats import norm

def calc_prior(X, m, n, cl1, cl2):
    return ((X == cl1).sum()) / (m*n), ((X == cl2).sum()) / (m*n) # p(cl1), p(cl2) estimates

def MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    flat_Y = Y.reshape(-1,1)
    norm_1 = norm(m1, sig1)
    norm_2 = norm(m2, sig2)
    
    Y_hat = np.array([cl1 if p1*norm_1.pdf(y-cl1) > p2*norm_2.pdf(y-cl2) else cl2 for y in flat_Y])
    
    return Y_hat.reshape(Y.shape)

def calc_posterior_Gauss(Y, m, n, p1, p2, m1, sig1, m2, sig2):
    pass

def main():
    path = './images_BW/cible2.bmp'
    X, shape = read_image(path)
    display_image('X (press any key to close window)', X * 255) # displays X
    
    cl1, cl2 = identify_classes(X) # identify image classes (two in our case)
    print(f'Class 1 : {cl1} (black)\nClass 2 : {cl2} (white)') 
    
    m, n = shape[0], shape[1] # X shape
    
    p1, p2 = calc_prior(X, m, n, cl1, cl2) # priors of each class
    print(f'\np(cl1) = p1 = {p1}\np(cl2) = p2 = {p2}')
    
    m1, m2, sig1, sig2 = 0, 0, .6, .6 # gaussian noise parameters    
    Y = X + gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2) # adding noise to image
    display_image('Y (press any key to close window)', Y)
    
    # Hocus pocus, you lost your focus ! and forgot everything you knew about the above parameters
    p1, p2, m1, m2, sig1, sig2 = 0, 0, 0, 0, 0, 0
    
    Y_hat = MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    display_image('Y_hat (press any key to close window)', Y_hat*255)
    
    print(f'\nError rate MPM : {error_rate(0, X, Y_hat, m, n):.1%}')
    print(f'Error rate KMeans : {evaluate_kmeans(2, X, Y)[1]:.1%}')
    
main()
    

    