import basictools as bt
import numpy as np

def make_table(paths, table=[]):
    
    for index,path in enumerate(paths):
        print(f'Image : {path.split("/")[-1]}')
    
        image, shape = bt.read_image(path)
        cl1, cl2 = bt.identify_classes(image)
        print(f'classe 1 : {cl1}, classe 2 : {cl2}')
    
        noise_1_1 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 0, 1, 0, 0)
        noise_1_2 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 0, 0, 0, 1)
        noise_2_1 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 1, 1, 0, 0)
        noise_2_2 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 1, 1, 0, .3)
        noise_3_1 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 0, .3, -1, .5)
        noise_3_2 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 0, .2, 0, .2)

        table.append([image])
        table[index].append(image + noise_1_1)
        table[index].append(image + noise_1_2)
        table[index].append(image + noise_2_1)
        table[index].append(image + noise_2_2)
        table[index].append(image + noise_3_1)
        table[index].append(image + noise_3_2)
    
        table[index] = np.concatenate(tuple(table[index]), axis=1)
    
    final_table = np.concatenate(tuple(table), axis=0)
    return final_table


def main():
    paths = ['./images_BW/alfa2.bmp', './images_BW/beee2.bmp', './images_BW/cible2.bmp', './images_BW/city2.bmp', './images_BW/country2.bmp']
    table = make_table(paths)
    # bt.display_image('Comparative table', table)
    
    im_width = 256
    elements_per_row = 7
    # code for row, just gotta extract method and iterate over rows
    row_1 = table[:im_width, :]
    kmeans_results = []
    
    for i in range(elements_per_row):
        kmeans_results.append(bt.evaluate_kmeans(2, row_1[:,:im_width], row_1[:, (i)*im_width:(i+1)*im_width]))
    
    for index, error in enumerate(kmeans_results):
        print(f'\n{index}) Error rate between image & KMeans : {error[1]:.1%}')
        print(f'   Error rate between image & Noisy :  {bt.error_rate(0.1, row_1[:,:im_width], row_1[:, (index)*im_width:(index+1)*im_width], im_width, im_width):.1%}')
    
    kmeans_images = np.concatenate(tuple([img[0] for img in kmeans_results]), axis=1)
    bt.display_image('Kmeans on noisy', np.concatenate((row_1, kmeans_images)))

main()

# parametre kmeans pour multiple initialisations + enlever random_state fixe
# changer les valeurs des bruits pour avoir des résultats plus exploitables pour kmeans
# denoises for which kind of noises ?