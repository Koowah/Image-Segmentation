from codes.python import *
import cv2
import os

m, n = 256, 256 # image size

# calculates the conditional probabilities of each value of X (hidden markov field) given the number of similar neighbours for a 4 neighbours configuration
proba_Markov_field = calc_proba_champ(alpha=1) # array where element (i,j) = P[xs = ωj | i+1 neighbours with class ωj]

# classes of the image
classes = [0,1]

# this function uses gibbs sampling to generate X from its conditional distributions
X = genere_Gibbs_proba(m + 2, n + 2, classes, proba_Markov_field, 100)
X = X.astype('uint8') # necessary to display image with cv2.imshow - initial type int32
X = redecoupe_image(X) # gets rid of first & last rows + columns that can't be updated by Gibbs

assert(X.shape == (256,256), 'Shape should be 256*256')

filename = 'Hidden_markov_field.png'
cv2.imwrite(filename, X*255) # write X into an image file

# To observe the effects each gibbs iteration has over image, check out genere_Gibbs_proba
# function in codes/python.py and uncomment relevant lines