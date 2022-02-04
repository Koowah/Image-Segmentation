from basictools import *
from scipy.stats import norm

def calc_prior(X, m, n, cl1, cl2):
    return ((X == cl1).sum()) / (m*n), ((X == cl2).sum()) / (m*n) # p(cl1), p(cl2) estimates


def MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    flat_Y = Y.reshape(-1,1)
    norm_1 = norm(m1, sig1)
    norm_2 = norm(m2, sig2)
    
    condition = p1*norm_1.pdf(flat_Y-cl1) > p2*norm_2.pdf(flat_Y-cl2)
    
    Y_hat = cl1 * condition + cl2 * (np.invert(condition))
    
    # Y_hat = np.array([cl1 if p1*norm_1.pdf(y-cl1) > p2*norm_2.pdf(y-cl2) else cl2 for y in flat_Y]) # less numpy friendly way to put it
    # takes SO much more time to execute !
    
    return Y_hat.reshape(Y.shape)


def main():
    path = './images_BW/cible2.bmp'
    X, shape = read_image(path)
    display_image('X', X * 255) # displays X
    
    cl1, cl2 = identify_classes(X) # identify image classes (two in our case)
    print(f'Class 1 : {cl1} (black)\nClass 2 : {cl2} (white)') 
    
    m, n = shape[0], shape[1] # X shape
    
    p1, p2 = calc_prior(X, m, n, cl1, cl2) # priors of each class
    print(f'\np(cl1) = p1 = {p1}\np(cl2) = p2 = {p2}')
    
    m1, m2, sig1, sig2 = .3, 0, .6, .6 # gaussian noise parameters
    Y = X + gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2) # adding noise to image
    display_image('Y', Y)
    
    Y_hat = MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    display_image('Y_hat', Y_hat*255)
    
    print(f'\nError rate MPM : {error_rate(0, X, Y_hat, m, n):.1%}')
    print(f'Error rate KMeans : {evaluate_kmeans(2, X, Y)[1]:.1%}')
  

if __name__ == '__main__':  
    main()
    

    