from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def model(input_shape):
    
    model = Sequential()
    
    model.add(Conv2D(32, 
                     kernel_size = (3,3),
                     activation = "relu",
                     kernel_initializer = "he_normal",
                     input_shape = input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,
                     kernel_size = (3,3),
                     activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128,
                     kernel_size = (3,3),
                     activation = "relu"))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(128,
                    activation = "relu"))
    model.add(Dropout(0.3))
    
    model.add(Dense(1,
                    activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy",
                  optimizer = "adam",
                  metrics = ["accuracy"])
    
    return model


if __name__ == "__main__":
    import argparse
    from utils import check_tuple
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type = tuple, default = (100,100))
    parser.add_argument("--num_channels", type = int, default = 3)
    parser.add_argument("-smp", "--save_model_path", type = str, default = "models/ClothesModel_1.h5")
    parser.add_argument("--print_summary", type = bool, default = True)
    
    args = parser.parse_args()
    
    RESOLUTION = check_tuple(args.resolution, (100,100))
    INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], args.num_channels)#(100,100,3)
    
    model = model(INPUT_SHAPE)
    if (args.print_summary): model.summary()
    model.save(args.save_model_path)
    print("Saved.")