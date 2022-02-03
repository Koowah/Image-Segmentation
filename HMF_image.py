from fileinput import filename
from codes.python import *
import cv2

m, n = 256, 256 # image size

# calculates the conditional probabilities of each value of X (hidden markov field) given the number of similar neighbours for a 4 neighbours configuration
proba_HMF = calc_proba_champ(alpha=1) # array where element (i,j) = P[xs = ωj | i+1 neighbours with class ωj]

# classes of the image
classes = [0,1]

# this function uses gibbs sampling to generate X from its conditional distributions
X = genere_Gibbs_proba(m + 2, n + 2, classes, proba_HMF, 50)
X = X.astype('uint8') # necessary to display image with cv2.imshow - initial type int32
X = redecoupe_image(X) # gets rid of first & last rows + columns that can't be updated by Gibbs

assert(X.shape == (256,256), 'Shape should be 256*256')

filename = 'Hidden_markov_field.png'
cv2.imwrite(filename, X*255) # write X into an image file

# To assess the effects nb gibbs iterations
list_HMF = {}
for i in range(1, 10):
    list_HMF[f'X_{5*i}'] = genere_Gibbs_proba(m+2, n+2, classes, proba_HMF, 5*i)
    list_HMF[f'X_{5*i}'] = list_HMF[f'X_{5*i}'].astype('uint8')
    list_HMF[f'X_{5*i}'] = redecoupe_image(list_HMF[f'X_{5*i}'])
    
list_HMF['X_200'] = genere_Gibbs_proba(m+2, n+2, classes, proba_HMF, 200)
list_HMF['X_200'] = list_HMF['X_200'].astype('uint8')
list_HMF['X_200'] = redecoupe_image(list_HMF['X_200'])

# write all images in files
for item in list_HMF.items():
    name = item[0]
    image = item[1]
    
    cv2.imwrite(name + '.png', image*255) # write X into an image file