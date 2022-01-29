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
    
    return Y_hat.reshape(Y.shape)


def calc_posterior_Gauss(Y, m, n, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    # posterior = p(X | Y) = p(Y|X)p(X) / sum(p(Y|X)p(X)) for each class X
    
    flat_Y = Y.reshape(-1,1)
    norm_1 = norm(m1, sig1)
    norm_2 = norm(m2, sig2)
    
    f1 = p1*norm_1.pdf(flat_Y-cl1) # this way we call stats.norm only once
    f2 = p2*norm_2.pdf(flat_Y-cl2)
    
    post_cl1 = (f1 / (f1 + f2))
    post_cl2 = (f2 / (f1 + f2))
    
    # posterior = np.concatenate((post_cl1, post_cl2))
    
    # assert np.prod(posterior.shape) == m*n*2, 'posterior shape should be m*n*2'
    
    return post_cl1, post_cl2


def calc_EM(Y, m, n, cl1, cl2, p10, p20, m10, sig10, m20, sig20, nb_iterEM):
    Y_flat = Y.reshape(-1,1)
    
    for i in range(nb_iterEM):
        # Expectation
        P1, P2 = calc_posterior_Gauss(Y, m, n, cl1, cl2, p10, p20, m10, sig10, m20, sig20)
       
        # Maximization
        p10 = P1.sum()/len(P1)
        p20 = P2.sum()/len(P2)
        m10 = (Y_flat*P1).sum()/P1.sum()
        sig10 = ((Y_flat - m10)*P1).sum()/P1.sum()
        m20 = (Y_flat*P2).sum()/P2.sum()
        sig20 = ((Y_flat - m20)*P2).sum()/P2.sum()
    
    return p10, p20, m10, sig10, m20, sig20


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
    # p1, p2, m1, m2, sig1, sig2 = 0, 0, 0, 0, 0, 0
    
    Y_hat = MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    display_image('Y_hat (press any key to close window)', Y_hat*255)
    
    print(f'\nError rate MPM : {error_rate(0, X, Y_hat, m, n):.1%}')
    print(f'Error rate KMeans : {evaluate_kmeans(2, X, Y)[1]:.1%}')
    
main()
    

    