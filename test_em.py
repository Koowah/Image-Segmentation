from basictools import *
from scipy.stats import norm

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


###################################################################################
###################################################################################
###################################################################################


def calc_prior(X, m, n, cl1, cl2):
    return ((X == cl1).sum()) / (m*n), ((X == cl2).sum()) / (m*n) # p(cl1), p(cl2) estimates

def calc_posterior_Gauss(Y, m, n, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    # posterior = p(X | Y) = p(Y|X)p(X) / sum(p(Y|X)p(X)) for each class X
    
    flat_Y = Y.reshape(-1,1)
    norm_1 = norm(m1, sig1)
    norm_2 = norm(m2, sig2)
    
    f1 = p1*norm_1.pdf(flat_Y - cl1) # this way we call stats.norm only once
    f2 = p2*norm_2.pdf(flat_Y - cl2)
    
    post_cl1 = (f1 / (f1 + f2))
    post_cl2 = (f2 / (f1 + f2))
    
    # posterior = np.concatenate((post_cl1, post_cl2))
    
    # assert np.prod(posterior.shape) == m*n*2, 'posterior shape should be m*n*2'
    
    return post_cl1, post_cl2

def calc_EM(Y, m, n, cl1, cl2, p10, p20, m10, sig10, m20, sig20, nb_iterEM):
    Y_flat = Y.reshape(-1,1)
    dic = {'p1':[p10], 'p2':[p20], 'm1':[m10], 'm2':[m20], 'sig1':[sig10], 'sig2':[sig20]}
    
    for i in range(nb_iterEM):
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
    
    m1, m2, sig1, sig2 = .12, .06, .2, .2 # gaussian noise parameters    
    Y = X + gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2) # adding noise to image
    
    # Y = Y.clip(0,1) # clipping yields terrible results here as it denaturalizes the pdf
    # to put in perspective with the benefits of clipping for KMeans where we don't care about distribution
    # but only about proximity hence the interest of aggregating values under 0 and above 1 
    # which have no visual representation anyway
    
    display_image('Y', Y)
    
    # Hocus pocus, you lost your focus ! and forgot everything you knew about the above parameters
    # p1, p2, m1, m2, sig1, sig2 = 0, 0, 0, 0, 0, 0
    
    p1_em, p2_em, m1_em, sig1_em, m2_em, sig2_em, dic = calc_EM(Y, m, n, cl1, cl2, .2, .3, .0, .1, .02, .1, 10)
    print(f'p1: {p1_em}, p2 : {p2_em}, m1 : {m1_em}, sig1 : {sig1_em}, m2 : {m2_em}, sig2 : {sig2_em}')
    
    df = pd.DataFrame.from_dict(dic)
    
    sns.set()
    sns.relplot(data=df, kind='line')
    # uncomment lines below to plot each variable separately
    # for column in df.columns:
    #     sns.relplot(data=df[column], kind='line', palette='crest')
    plt.show()
    

if __name__ == '__main__':  
    main()
    

    