from random import gauss
from basictools import *
from scipy.stats import norm

def calc_prior(X, m, n, cl1, cl2):
    return ((X == cl1).sum()) / (m*n), ((X == cl2).sum()) / (m*n) # p(cl1), p(cl2) estimates


def MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    flat_Y = Y.reshape(-1,1)
    norm_1 = norm(m1, sig1)
    norm_2 = norm(m2, sig2)
    
    condition = p1*norm_1.pdf(flat_Y-cl1) >= p2*norm_2.pdf(flat_Y-cl2)
    
    X_hat = cl1 * condition + cl2 * (np.invert(condition))
    
    # Y_hat = np.array([cl1 if p1*norm_1.pdf(y-cl1) > p2*norm_2.pdf(y-cl2) else cl2 for y in flat_Y]) # less numpy friendly way to put it
    # takes SO much more time to execute !
    
    return X_hat.reshape(Y.shape)


def main():
    path = './images_BW/country2.bmp'
    X, shape = read_image(path)
    display_image('X', X * 255) # displays X
    
    cl1, cl2 = identify_classes(X) # identify image classes (two in our case)
    print(f'Class 1 : {cl1} (black)\nClass 2 : {cl2} (white)') 
    
    m, n = shape[0], shape[1] # X shape
    
    p1, p2 = calc_prior(X, m, n, cl1, cl2) # priors of each class
    print(f'\np(cl1) = p1 = {p1}\np(cl2) = p2 = {p2}')
    
    # noises we'll be using
    noise_1 = gauss_noise(X, shape[0], shape[1],cl1, cl2, .3, .1, -.4, .1)
    noise_2 = gauss_noise(X, shape[0], shape[1],cl1, cl2, .3, .3, -.55, .3)
    noise_3 = gauss_noise(X, shape[0], shape[1],cl1, cl2, .4, .3, -.6, .47)

    # adding gaussian noise to our image
    noisy_1 = X + noise_1
    noisy_2 = X + noise_2
    noisy_3 = X + noise_3
        
    # MPM on noisy_1 & error evaluation
    m1, sig1, m2, sig2 = .3, .1, -.4, .1
    X_hat_1 = MPM_Gauss(noisy_1, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    
    MPM_result_1 = np.concatenate((X, noisy_1, X_hat_1), axis=1)
    display_image('Image, Noisy_1, MPM', MPM_result_1)
    print(f'\nError rate MPM noisy_1 : {error_rate(0, X, X_hat_1, m, n):.2%}')
    
    # cv2.imwrite('MPM_result_1.png', 255 * MPM_result_1) # write result into PNG file

    # Kmeans on noisy_2 & error evaluation
    m1, sig1, m2, sig2 = .3, .3, -.55, .3
    X_hat_2 = MPM_Gauss(noisy_2, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    
    MPM_result_2 = np.concatenate((X, noisy_2, X_hat_2), axis=1)
    display_image('Image, Noisy_2, KMeans', MPM_result_2)
    print(f'\nError rate MPM noisy_2 : {error_rate(0, X, X_hat_2, m, n):.2%}')
    
    # cv2.imwrite('MPM_result_2.png', 255 * MPM_result_2)
    
    # Kmeans on noisy_3 & error evaluation
    m1, sig1, m2, sig2 = .4, .3, -.6, .47
    X_hat_3 = MPM_Gauss(noisy_3, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    
    MPM_result_3 = np.concatenate((X, noisy_3, X_hat_3), axis=1)
    display_image('Image, Noisy_2, KMeans', MPM_result_3)
    print(f'\nError rate MPM noisy_3 : {error_rate(0, X, X_hat_3, m, n):.2%}')
    
    # cv2.imwrite('MPM_result_3.png', 255 * MPM_result_3)
    
    
    # m1, sig1, m2, sig2 = .4, .3, -.6, .47 # gaussian noise parameters
    # Y = X + gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2) # adding noise to image
    # display_image('Y', Y)
    
    # Y_hat = MPM_Gauss(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2)
    # display_image('Y_hat', Y_hat*255)
    
    # print(f'\nError rate MPM : {error_rate(0, X, Y_hat, m, n):.1%}')
  

if __name__ == '__main__':  
    main()
    

    