from basictools import *
from scipy.stats import norm

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


###################################################################################
###################################################################################
###################################################################################


def calc_prior(X, m, n, cl1, cl2):
    return ((X == cl1).sum()) / (m*n), ((X == cl2).sum()) / (m*n) # p(cl1), p(cl2) estimates

def calc_posterior_Gauss(Y, m, n, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    # posterior = p(X | Y) = p(Y|X)p(X) / sum(p(Y|X)p(X)) for each class X
    flat_Y = Y.reshape(-1,1)

    if (p1, p2, m1, sig1, m2, sig2) == (0, 0, 0, 0, 0, 0):
        post_cl1 = np.invert(flat_Y)
        post_cl2 = flat_Y
    else:
        norm_1 = norm(m1, sig1)
        norm_2 = norm(m2, sig2)
        
        f1 = p1*norm_1.pdf(flat_Y - cl1) # this way we call stats.norm only once
        f2 = p2*norm_2.pdf(flat_Y - cl2)
        
        post_cl1 = (f1 / (f1 + f2))
        post_cl2 = (f2 / (f1 + f2))
        
        # posterior = np.concatenate((post_cl1, post_cl2))
        # assert np.prod(posterior.shape) == m*n*2, 'posterior shape should be m*n*2'
    
    
    return post_cl1, post_cl2

def posterior_sampling(post_cl1, post_cl2, cl1, cl2, m, n): # tirage_apost
    sample = np.random.rand(m*n).reshape(-1,1) # m*n samples from uniform distribution over [0,1)
    
    condition1 = sample < post_cl1 # we draw cl1 with posterior probability of cl1 aka when our uniform sample(i) < post_cl1(i) and cl2 otherwise
    condition2 = sample < post_cl2
    X_est = (cl1 * condition1) + (cl2 * condition2)
    
    return X_est

def init_param(Y, cl1, cl2): # initialize an estimate of X from KMeans
    X_init, _ = evaluate_kmeans(2, Y, Y)
    X_init[X_init == np.unique(X_init)[0]] = cl1
    X_init[X_init == np.unique(X_init)[1]] = cl2
    
    return X_init

def est_empirical(X, Y, cl1, cl2): # Empirical estimates of parameters
    Y_flat = Y.reshape(-1,1)
    X_flat = X.reshape(-1,1)
    
    p1 = (X_flat == cl1).sum() / np.prod(X_flat.shape) # frequency estimates
    p2 = (X_flat == cl2).sum() / np.prod(X_flat.shape)
    m1 = (Y_flat[X_flat == cl1] - cl1).sum() / (X_flat == cl1).sum() # means over noise realizations
    m2 = (Y_flat[X_flat == cl2] - cl2).sum() / (X_flat == cl2).sum()
    sig1 = np.sqrt(((Y_flat[X_flat == cl1] - m1)**2).sum() / (X_flat == cl1).sum()) # stds over noise realizations
    sig2 = np.sqrt(((Y_flat[X_flat == cl2] - m2)**2).sum() / (X_flat == cl2).sum())
    
    return p1, p2, m1, sig1, m2, sig2

def calc_SEM(Y, m, n, cl1, cl2, p10, p20, m10, sig10, m20, sig20, nb_iter):
    # Y_flat = Y.reshape(-1,1)
    dic = {'p1':[p10], 'p2':[p20], 'm1':[m10], 'm2':[m20], 'sig1':[sig10], 'sig2':[sig20]}
    
    for _ in range(nb_iter):
        # Calculate posterior with current parameters
        post1, post2 = calc_posterior_Gauss(Y, m, n, cl1, cl2, p10, p20, m10, sig10, m20, sig20)
        
        # Sample X from posterior distribution
        X_est = posterior_sampling(post1, post2, cl1, cl2, m, n)
        
        # Empirically estimates of parameters from X_est
        p10, p20, m10, sig10, m20, sig20 = est_empirical(X_est, Y, cl1, cl2)
        
        # To plot convergence
        dic['p1'].append(p10)
        dic['p2'].append(p20)
        dic['m1'].append(m10)
        dic['m2'].append(m20)
        dic['sig1'].append(sig10)
        dic['sig2'].append(sig20)
    
    return p10, p20, m10, sig10, m20, sig20, dic

def calc_EM(Y, m, n, cl1, cl2, p10, p20, m10, sig10, m20, sig20, nb_iterEM):
    Y_flat = Y.reshape(-1,1)
    dic = {'p1':[p10], 'p2':[p20], 'm1':[m10], 'm2':[m20], 'sig1':[sig10], 'sig2':[sig20]}
    
    for _ in range(nb_iterEM):
        # Expectation
        post1, post2 = calc_posterior_Gauss(Y, m, n, cl1, cl2, p10, p20, m10, sig10, m20, sig20)
       
        # Maximization
        p10 = post1.sum()/len(post1)
        p20 = post2.sum()/len(post2)
        m10 = (Y_flat*post1).sum()/post1.sum()
        sig10 = np.sqrt(((((Y_flat - m10)**2) * post1).sum() / post1.sum()))
        m20 = ((Y_flat - cl2)*post2).sum()/post2.sum()
        sig20 = np.sqrt((((((Y_flat - cl2) - m20)**2) * post2).sum() / post2.sum()))
        
        # To plot convergence
        dic['p1'].append(p10)
        dic['p2'].append(p20)
        dic['m1'].append(m10)
        dic['m2'].append(m20)
        dic['sig1'].append(sig10)
        dic['sig2'].append(sig20)
    
    return p10, p20, m10, sig10, m20, sig20, dic


###################################################################################
###################################################################################
###################################################################################


def main():
    path = './images_BW/cible2.bmp'
    X, shape = read_image(path)
    display_image('X', X * 255) # displays X
    
    cl1, cl2 = identify_classes(X) # identify image classes (two in our case)
    print(f'Class 1 : {cl1} (black)\nClass 2 : {cl2} (white)') 
    
    m, n = shape[0], shape[1] # X shape
    
    p1, p2 = calc_prior(X, m, n, cl1, cl2) # priors of each class
    print(f'\np(cl1) = p1 = {p1}\np(cl2) = p2 = {p2}')
    
    m1, sig1, m2, sig2 = .2, .2, -.3, .2 # gaussian noise parameters    
    Y = X + gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2) # adding noise to image
    display_image('Y', Y)
    
    ################################################################################################
    # Hocus pocus, you lost your focus ! and forgot everything you knew about the above parameters #
    ################################################################################################
    
    p1, p2, m1, m2, sig1, sig2 = 0, 0, 0, 0, 0, 0 # he 4gett ... but he also estimett !
    p1, p2, m1, sig1, m2, sig2 = est_empirical(init_param(Y, cl1, cl2), Y, cl1, cl2) # empirically estimating a starting point for EM & SEM algorithms
    
    # p1_sem, p2_sem, m1_sem, sig1_sem, m2_sem, sig2_sem, dic_sem = calc_SEM(Y, m, n, cl1, cl2, p1, p2, m1, sig1, m2, sig2, 10)
    p1_em, p2_em, m1_em, sig1_em, m2_em, sig2_em, dic_em = calc_EM(Y, m, n, cl1, cl2, p1, p2, m1, sig1, m2, sig2, 300) # EM algorithm estimates
    
    print(f'EM\np1: {p1_em}, p2 : {p2_em}, m1 : {m1_em}, sig1 : {sig1_em}, m2 : {m2_em}, sig2 : {sig2_em}')
    # print(f'SEM\np1: {p1_sem}, p2 : {p2_sem}, m1 : {m1_sem}, sig1 : {sig1_sem}, m2 : {m2_sem}, sig2 : {sig2_sem}')
    
    df_em = pd.DataFrame.from_dict(dic_em)
    # df_sem = pd.DataFrame.from_dict(dic_sem)
    
    sns.set()
    sns.relplot(data=df_em, kind='line') # Plotting evolution of parameters over EM iterations
    # sns.relplot(data=df_sem, kind='line') # Plotting evolution of parameters over SEM iterations
    plt.show()

# si le bruit 0, 0, .6, .6 ça converge vers un minimum local        
main()
    

    