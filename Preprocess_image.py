from os import listdir
import numpy as np
import cv2
from utils import check_tuple

def preprocess_image2(input_image_path, output_image_path, model_image_size, grayscale = False, save = False, resize = True):
    
    image = cv2.imread(input_image_path, 0 if (grayscale) else 1)
    if resize:
        resized_image = cv2.resize(image, (model_image_size[1], model_image_size[0]))#if resize else image
    else:
        resized_image = image
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.

    if (grayscale): resized_image = np.expand_dims(resized_image, 3)# Add last axis
    image_data = np.expand_dims(resized_image, 0)  # Add batch dimension.
        
    if (save):
        cv2.imwrite(output_image_path, resized_image)
        
    return image_data
    
    
def take_n_resize_images(input_path, output_path = "", image_size=(100,100), num_examples=None, grayscale = False,   save = False, resize = True, start = 0):
    image_size = check_tuple(image_size, (100,100))
    if num_examples == None: num_examples = len(listdir(input_path))
    images = np.zeros((num_examples,
                       image_size[0],
                       image_size[1],
                       1 if grayscale else 3))
    for i,file in enumerate(listdir(input_path)):
        if i < start: continue
        if i == start+num_examples: break
        images[i-start] = preprocess_image2(input_path +'/'+file, 
                                     output_path +'/'+file, 
                                     image_size,
                                     grayscale,
                                     save,
                                     resize)


    return images

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str, default = "../Datasets/datasets/background/test")
    parser.add_argument("--save_path", type = str, default = "")
    parser.add_argument("--resolution", type = tuple, default = (100,100))
    parser.add_argument("--num_examples", type = int, default = 1)
    parser.add_argument("--start", type = int, default = 0)
    parser.add_argument("--grayscale", type = bool, default = False)
    parser.add_argument("--save", type = bool, default = False)
    parser.add_argument("--resize", type = bool, default = True)
    
    args = parser.parse_args()
    imgs = take_n_resize_images(args.data_path,
                                output_path = args.save_path,
                                image_size = args.resolution,
                                num_examples = args.num_examples,
                                start = args.start,
                                grayscale = args.grayscale,
                                save = args.save,
                                resize = args.resize)
    print("Images (shape): ", imgs.shape)