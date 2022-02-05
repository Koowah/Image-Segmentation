from codes.python import *
from basictools import*

def calc_prior(X, m, n, cl1, cl2):
    return ((X == cl1).sum()) / (m*n), ((X == cl2).sum()) / (m*n) # p(cl1), p(cl2) estimates

def main():
    # Let's gain time by using previously generated Markov field
    path_markov_field = './Markov_field_sample_1000_gibbs.png'
    proba_Markov_field = calc_proba_champ(alpha=1)

    X, shape = read_image(path_markov_field)
    display_image('Markov field X', 255 * X)

    cl1, cl2 = identify_classes(X)
    classes = [cl1, cl2]
    print(f'Class 1 : {cl1} (black)\nClass 2 : {cl2} (white)') 
        
    m, n = shape[0], shape[1] # X shape
    print(f'Shape : {m}*{n}')
        
    p1, p2 = calc_prior(X, m, n, cl1, cl2) # priors of each class
    print(f'\np(cl1) = p1 = {p1}\np(cl2) = p2 = {p2}')

    m1, sig1, m2, sig2 = .3, .6, 0, .6 # gaussian noise parameters
    Y = X + gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2) # adding noise to image
    display_image('Noisy Markov field Y', Y)
    
    Y_prep = nouvelle_image(Y) # adds 2 additionnal rows & columns so that gibbs sampler runs on full image Y
    display_image('Noisy Markov field Y + new rows/cols', Y_prep)
    
    X_hat = MPM_proba_gauss(Y_prep, classes, m1, sig1, m2, sig2, proba_Markov_field, 30, 3) # tried w/ 200 gibbs not good
    X_hat = redecoupe_image(X_hat)
    display_image('MPM estimated X', 255*X_hat)
    
    filename = 'HMF_MPM.png'
    cv2.imwrite(filename, X_hat*255) # write X into an image file
    
    # assert(X_hat.shape == (m,n), 'Shape should be 256*256')
    
    error = error_rate(0, X, X_hat, m, n)
    print(f'MPM error rate for HMF : {error}')

if __name__ == '__main__':
    main()