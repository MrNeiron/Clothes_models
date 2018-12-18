import numpy as np
from Preprocess_image import take_n_resize_images as take
from keras.models import load_model
from sklearn.model_selection import train_test_split
import argparse
from utils import check_tuple

#                                            PARSING
parser = argparse.ArgumentParser()
parser.add_argument("-tpp", "--train_positive_path", type = str, default = "../Datasets/datasets/clothes_n_shoes/train2500")
parser.add_argument("-tnp", "--train_negative_path", type = str, default = "../Datasets/datsets/background/train2500")
parser.add_argument("--offPos", type = bool, default = False)
parser.add_argument("--offNeg", type = bool, default = False)
parser.add_argument("-np", "--num_positive", type = int, default = 1)
parser.add_argument("-nn", "--num_negative", type = int, default = 1)
parser.add_argument("-mp", "--model_path", type = str, default = "models/ClothesModel_2_pretrained.h5")
parser.add_argument("-smp", "--save_model_path", type = str, default = "models/ClothesModel_2_pretrained.h5")
parser.add_argument("-sp", "--start_positive", type = int, default = 0)
parser.add_argument("-sn", "--start_negative", type = int, default = 0)
parser.add_argument("--testOn", type = bool, default = False)

parser.add_argument("--test_p_path", type = str, default = "../Datasets/datasets/clothes_n_shoes/test")
parser.add_argument("--test_n_path", type = str, default = "../Datasets/datasets/background/test")
parser.add_argument("--num_positive_test", type = int, default = 10)
parser.add_argument("--num_negative_test", type = int, default = 10)
parser.add_argument("--start_positive_test", type = int, default = 0)
parser.add_argument("--start_negative_test", type = int, default = 0)

parser.add_argument("-vs", "--validation_size", type = float, default = 0.2)
parser.add_argument("--random_state", type = int, default = 2018)
parser.add_argument("--resolution", type = tuple, default = (100,100))
parser.add_argument("--grayscale", type = bool, default = False)

parser.add_argument("-bs", "--batch_size", type = int, default = 200)
parser.add_argument("-e", "--epochs", type = int, default = 3)

args = parser.parse_args()
    
if (args.offPos): args.num_positive = 0
if (args.offNeg): args.num_negative = 0

#                                                 PREPROCESS    
VALIDATION_SIZE = args.validation_size      #0.2
RANDOM_STATE = args.random_state            #2018
RESOLUTION = args.resolution                #(100,100)
GRAYSCALE = args.grayscale                  #False

BATCH_SIZE = args.batch_size                #200
EPOCHS = args.epochs                        #3

ClothesModel = load_model(args.model_path)

#                                                       TRAIN

X_clothes_n_shoes = take(args.train_positive_path,
                         image_size = RESOLUTION,
                         grayscale = GRAYSCALE,
                         num_examples = args.num_positive,
                         start = args.start_positive)

X_others = take(args.train_negative_path,
                image_size = RESOLUTION,
                grayscale = GRAYSCALE,
                num_examples = args.num_negative,
                start = args.start_negative)


X_input = np.vstack((X_clothes_n_shoes,X_others))

print("X input(shape): ", X_input.shape)

Y_clothes_n_shoes = np.ones((X_clothes_n_shoes.shape[0], 1), dtype = np.int32)
Y_others = np.zeros((X_others.shape[0], 1), dtype = np.int32)

Y_input = np.vstack((Y_clothes_n_shoes,Y_others))

print("Y input(shape): ", Y_input.shape)

X_train, X_val, Y_train, Y_val = train_test_split(X_input, Y_input, test_size = VALIDATION_SIZE, random_state = RANDOM_STATE)

print("X train(shape): ", X_train.shape)
print("X validation(shape): ", X_val.shape)
print("Y train(shape): ", Y_train.shape)
print("Y validation(shape): ", Y_val.shape)


ClothesModel.fit(X_train, Y_train,
             batch_size = BATCH_SIZE,
             epochs = EPOCHS,
             verbose = 1,
             validation_data = (X_val, Y_val))

ClothesModel.save(args.save_model_path)

#                                                        TEST    
if (args.testOn):
    X_clothes_n_shoes_test = take(args.test_p_path,
                                  image_size = RESOLUTION,
                                  grayscale = GRAYSCALE,
                                  num_examples = args.num_positive_test,
                                  start = args.start_positive_test)

    X_others_test = take(args.test_n_path,
                         image_size = RESOLUTION,
                         grayscale = GRAYSCALE,
                         num_examples = args.num_negative_test,
                         start = args.start_negative_test)


    X_test = np.vstack((X_clothes_n_shoes_test,X_others_test))

    print("X test(shape): ", X_test.shape)

    Y_clothes_n_shoes_test = np.ones((X_clothes_n_shoes_test.shape[0], 1), dtype = np.int32)
    Y_others_test = np.zeros((X_others_test.shape[0], 1), dtype = np.int32)

    Y_test = np.vstack((Y_clothes_n_shoes_test,Y_others_test))

    print("Y test(shape): ", Y_test.shape)

    _, accuracy = ClothesModel.evaluate(X_test,
                                  Y_test,
                                  verbose=1)
    print("Test accuracy: ", accuracy)

