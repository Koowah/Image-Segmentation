from basictools import *
import numpy as np


def main():
    
    path = './images_BW/country2.bmp'
    print(f'Image : {path.split("/")[-1]}')

    image, shape = read_image(path)
    cl1, cl2 = identify_classes(image)
    print(f'classe 1 : {cl1}, classe 2 : {cl2}')
    
    m, n = shape[0], shape[1]

    # noises we'll be using
    noise_1 = gauss_noise(image, m, n,cl1, cl2, .3, .1, -.4, .1)
    noise_2 = gauss_noise(image, m, n,cl1, cl2, .3, .3, -.55, .3)
    noise_3 = gauss_noise(image, m, n,cl1, cl2, .4, .3, -.6, .47)

    # adding gaussian noise to our image
    noisy_1 = image + noise_1
    noisy_2 = image + noise_2
    noisy_3 = image + noise_3
    
    # displaying noisy images together
    noisy = (noisy_1, noisy_2, noisy_3)
    combine_noisy = np.concatenate(noisy, axis=1)
    display_image('noisy_1, noisy_2 & noisy_3', combine_noisy)
    
    # Kmeans on noisy_1 & error evaluation
    img_seg_1, error_1 = evaluate_kmeans(2, image, noisy_1)
    KM_result_1 = np.concatenate((image, noisy_1, img_seg_1), axis=1)
    display_image('Image, Noisy_1, KMeans', KM_result_1)
    print(f'Error rate between image & KMeans on noisy : {error_1:.2%}')
    
    # cv2.imwrite('KM_result_1.png', 255 * KM_result_1) # write result into PNG file

    # Kmeans on noisy_2 & error evaluation
    img_seg_2, error_2 = evaluate_kmeans(2, image, noisy_2)
    KM_result_2 = np.concatenate((image, noisy_2, img_seg_2), axis=1)
    display_image('Image, Noisy_2, KMeans', KM_result_2)
    print(f'Error rate between image & KMeans on noisy_1 : {error_2:.2%}')
    
    # cv2.imwrite('KM_result_2.png', 255 * KM_result_2)
    
    # Kmeans on noisy_3 & error evaluation
    img_seg_3, error_3 = evaluate_kmeans(2, image, noisy_3)
    KM_result_3 = np.concatenate((image, noisy_3, img_seg_3), axis=1)
    display_image('Image, Noisy_2, KMeans', KM_result_3)
    print(f'Error rate between image & KMeans on noisy_1 : {error_3:.2%}')
    
    # cv2.imwrite('KM_result_3.png', 255 * KM_result_3)

    

if __name__ == '__main__':
    main()