from codes.python import *
from basictools import*
from aveugle_non_super_2classes import init_param, est_empirical

# for plotting purposes
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def calc_prior(X, m, n, cl1, cl2):
    return ((X == cl1).sum()) / (m*n), ((X == cl2).sum()) / (m*n) # p(cl1), p(cl2) estimates

def init_param_EM(Y, classe):
    cl1 = classe[0]
    cl2 = classe[1]
    X_KMeans = init_param(Y, cl1, cl2)
    _, _, m1, sig1, m2, sig2 = est_empirical(X_KMeans, Y, cl1, cl2) # _, _ <=> p1, p2
    
    N_part = calc_N_part(X_KMeans, classe)
    proba = N_part / N_part.sum(axis=1).reshape(-1,1)
        
    assert int(proba.sum().item()) == 5, 'Each row should sum to 1'
    
    return m1, sig1, m2, sig2, proba

def EM_Gibbs_Gauss(Y, classe, m1, sig1, m2, sig2, proba, nb_iter_Gibbs_EM, nb_simu_EM):
    for __ in range(nb_simu_EM):
        X_est = genere_Gibbs_proba_apost(Y, m1, sig1, m2, sig2, classe, proba, nb_iter_Gibbs_EM) # X sampled from posterior distribution using MCMC approach
        X_est_new = redecoupe_image(X_est)
        
        N_part = calc_N_part(X_est_new, classe)
        proba = N_part / N_part.sum(axis=1).reshape(-1,1)
        
    N_post = calc_N_post(X_est_new, classe).astype(float)
    table_voisins = [(0, -1), (-1, 0), (0, 1), (1, 0)]
    for i in range(1, X_est.shape[0] - 1):
        for j in range(1, X_est.shape[1] - 1):
            # on récupère la configuration du voisinage
            # i.e. on compte le nombre de voisins à la classe 1
            config = 0
            for v in table_voisins:
                config += (X_est[i + v[0], j + v[1]] == classe[0])
                distribution_locale_apost = proba[config].copy() # copy ! # we get our prior
                distribution_locale_apost[0] *= (1 / (np.sqrt(2 * np.pi) * sig1) # and multiply it by likelihood to get posterior
                    * np.exp(-0.5 * (Y[i, j] - classe[0] - m1) ** 2 / sig1 ** 2))
                distribution_locale_apost[1] *= (1 / (np.sqrt(2 * np.pi) * sig2)
                    * np.exp(-0.5 * (Y[i, j] - classe[1] - m2) ** 2 / sig2 ** 2))
                distribution_locale_apost /= np.sum(distribution_locale_apost)
                N_post[..., 0][i - 1, j - 1] = distribution_locale_apost[0]
                N_post[..., 1][i - 1, j - 1] = distribution_locale_apost[1]
    Ppost = N_post
        
    return proba, Ppost
        
def estim_param_Gauss_noise_EM(Y, classe, Ppost):
    Y_flat = redecoupe_image(Y).reshape(-1,1)
    
    post1 = Ppost[..., 0].reshape(-1,1)
    post2 = Ppost[..., 1].reshape(-1,1)
    cl1 = classe[0]
    cl2 = classe[1]
    
    m1 = ((Y_flat - cl1)*post1).sum()/post1.sum()
    sig1 = np.sqrt((((((Y_flat - cl1) - m1)**2) * post1).sum() / post1.sum()))
    m2 = ((Y_flat - cl2)*post2).sum()/post2.sum()
    sig2 = np.sqrt((((((Y_flat - cl2) - m2)**2) * post2).sum() / post2.sum()))
    
    return m1, sig1, m2, sig2

def EM_Gauss(Y, classe, nb_iter_EM, nb_iter_Gibbs_EM, nb_simu_EM):
    m1, sig1, m2, sig2, proba = init_param_EM(Y, classe)
    dic = {'m1':[m1], 'm2':[m2], 'sig1':[sig1], 'sig2':[sig2]}
    
    for _ in range(nb_iter_EM):
        # Expectation
        proba, Ppost = EM_Gibbs_Gauss(Y, classe, m1, sig1, m2, sig2, proba, nb_iter_Gibbs_EM, nb_simu_EM)
        
        # Maximization
        m1, sig1, m2, sig2 = estim_param_Gauss_noise_EM(Y, classe, Ppost)
    
        # To plot convergence        
        dic['m1'].append(m1)
        dic['m2'].append(m2)
        dic['sig1'].append(sig1)
        dic['sig2'].append(sig2)
        
    return m1, sig1, m2, sig2, dic

def main():
    # Let's gain time by using previously generated Markov field
    path_markov_field = './Markov_field_sample_1000_gibbs.png'

    X, shape = read_image(path_markov_field)
    display_image('Markov field X', 255 * X)

    cl1, cl2 = identify_classes(X)
    classes = [cl1, cl2]
    print(f'Class 1 : {cl1} (black)\nClass 2 : {cl2} (white)') 
        
    m, n = shape[0], shape[1] # X shape
    print(f'Shape : {m}*{n}')
        
    p1, p2 = calc_prior(X, m, n, cl1, cl2) # priors of each class
    print(f'\np(cl1) = p1 = {p1:.3f}\np(cl2) = p2 = {p2:.3f}')

    m1, sig1, m2, sig2 = .3, .6, -.19, .47 # gaussian noise parameters
    Y = X + gauss_noise(X, m, n, cl1, cl2, m1, sig1, m2, sig2) # adding noise to image
    print(f'\nm1 = {m1:.2f}, sig1 = {sig1:.2f}, m2 = {m2:.2f}, sig2 = {sig2:.2f}')
    display_image('Noisy Markov field Y', Y)
    
    Y_prep = nouvelle_image(Y) # adds 2 additionnal rows & columns so that gibbs sampler runs on full image Y
    display_image('Noisy Markov field Y + new rows/cols', Y_prep)
    
    nb_iter_EM, nb_iter_Gibbs_EM, nb_simu_EM = 25, 3, 5
    m1_EM, sig1_EM, m2_EM, sig2_EM, dic_EM = EM_Gauss(Y_prep, classes, nb_iter_EM, nb_iter_Gibbs_EM, nb_simu_EM)
    
    print(f'\nm1 : {m1_EM}, sig1 : {sig1_EM}, m2 : {m2_EM}, sig2 : {sig2_EM}')
    
    df_em = pd.DataFrame.from_dict(dic_EM)
    
    sns.set()
    sns.relplot(data=df_em, kind='line') # Plotting evolution of parameters over EM iterations
    plt.show()
    
    # assert X_hat.shape == (m,n), 'Shape should be 256*256'
    
    # error = error_rate(0, X, X_hat, m, n)
    # print(f'MPM error rate for HMF : {error}')

if __name__ == '__main__':
    main()