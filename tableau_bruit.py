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
        noise_2_2 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 0, 0, -1, .1)
        noise_3_1 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 1, 1, 0, .3)
        noise_3_2 = bt.gauss_noise(image, shape[0], shape[1],cl1, cl2, 0, 1, 0, 1)

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
    bt.display_image('Comparative table', table)

if __name__ == '__main__':
    main()