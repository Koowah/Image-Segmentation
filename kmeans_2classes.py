from basictools import *
import numpy as np


def main():
    
    path = './images_BW/cible2.bmp'
    print(f'Image : {path.split("/")[-1]}')

    image, shape = read_image(path)
    cl1, cl2 = identify_classes(image)
    print(f'classe 1 : {cl1}, classe 2 : {cl2}')

    # noisy (mean : 0, std_cl1 : 1) - noisy_1 (mean : 0, std_both_classes : 1)
    noisy = (image + gauss_noise(image, shape[0], shape[1],cl1, cl2, 0, .3, 0, 0))
    noisy_1 = (noisy + gauss_noise(image, shape[0], shape[1],cl1, cl2, 0, 0, 0, .3))
    noice = [noisy, noisy_1]
    combine_noisy = np.concatenate(tuple(noice), axis=1)
    display_image('Noisy, Noisy_1', combine_noisy)
    
    # Kmeans on noisy & error evaluation
    img_seg, error = evaluate_kmeans(2, image, noisy)
    display_image('Noisy, KMeans, Image', np.concatenate((noisy, img_seg, image), axis=1))
    print(f'Error rate between image & KMeans : {error:.1%}')
    print(f'Error rate between image & Noisy :  {error_rate(0.1, image, noisy, shape[0], shape[1]):.1%}')

    # Kmeans on noisy_1 & error evaluation
    img_seg, error = evaluate_kmeans(2, image, noisy_1)
    display_image('Noisy_1, KMeans, Image', np.concatenate((noisy_1, img_seg, image), axis=1))
    print(f'Error rate between image & KMeans : {error:.1%}')
    print(f'Error rate between image & Noisy_1 :  {error_rate(0.1, image, noisy_1, shape[0], shape[1]):.1%}')
    

if __name__ == '__main__':
    main()