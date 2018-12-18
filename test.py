import numpy as np
from Preprocess_image import take_n_resize_images
from keras.models import load_model
import argparse

#                                       PARSING
parser = argparse.ArgumentParser()
parser.add_argument("-tpp", "--test_p_path", type = str, default = "../Datasets/datasets/clothes_n_shoes/test")
parser.add_argument("-tnp", "--test_n_path", type = str, default = "../Datasets/datasets/background/test")
parser.add_argument("-np", "--num_positive", type = int, default = 10)
parser.add_argument("-nn", "--num_negative", type = int, default = 10)
parser.add_argument("-sp", "--start_positive", type = int, default = 0)
parser.add_argument("-sn", "--start_negative", type = int, default = 0)
parser.add_argument("-mp", "--model_path", type = str, default = "models/ClothesModel_2_pretrained.h5")

parser.add_argument("--resolution", type = tuple, default = (100,100))
parser.add_argument("--grayscale", type = bool, default = False)

args = parser.parse_args()

#                                       PARSING TEST DATA
X_clothes_n_shoes_test = take_n_resize_images(args.test_p_path,
                                               image_size = args.resolution,
                                               grayscale = args.grayscale,
                                               num_examples = args.num_positive,
                                               start = args.start_positive)

X_others_test = take_n_resize_images(args.test_n_path,
                                     image_size = args.resolution,
                                     grayscale = args.grayscale,
                                     num_examples = args.num_negative,
                                     start = args.start_negative)


X_test = np.vstack((X_clothes_n_shoes_test,X_others_test))

print("X test(shape): ", X_test.shape)

Y_clothes_n_shoes_test = np.ones((X_clothes_n_shoes_test.shape[0], 1), dtype = np.int32)
Y_others_test = np.zeros((X_others_test.shape[0], 1), dtype = np.int32)

Y_test = np.vstack((Y_clothes_n_shoes_test,Y_others_test))

print("Y test(shape): ", Y_test.shape)


#                                             TEST
ClothesModel = load_model(args.model_path)

_ , accuracy = ClothesModel.evaluate(X_test,
                              Y_test,
                              verbose=1)
print("Test2 accuracy: ", accuracy)

 