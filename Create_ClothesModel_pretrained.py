from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Input

def model(input_shape):
    vggModel = VGG16(weights = "imagenet", include_top = False)
    for layer in vggModel.layers:
        layer.trainable = False
        
    input_tensor = Input(shape = input_shape, name = "image_input")
    vggModel_output = vggModel(input_tensor)
    
    X = Flatten()(vggModel_output)
    X = Dense(1024, activation = "relu", name = "fc1")(X)
    X = Dense(1, activation = "sigmoid", name = "predictions")(X)
    
    model = Model(input = input_tensor, output = X)
    
    model.compile(loss = "binary_crossentropy",
               optimizer = "adam",
               metrics = ["accuracy"])
    
    return model
    
    
if __name__ == "__main__":
    import argparse
    from utils import check_tuple
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type = tuple, default = (100,100))
    parser.add_argument("-smp", "--save_model_path", type = str, default = "models/ClothesModel_2_pretrained.h5")
    parser.add_argument("--print_summary", type = bool, default = True)
    
    args = parser.parse_args()
    
    RESOLUTION = check_tuple(args.resolution, (100,100))
    INPUT_SHAPE = (RESOLUTION[0], RESOLUTION[1], 3)#(100,100,3)
    
    model = model(INPUT_SHAPE)
    if (args.print_summary): model.summary()
    model.save(args.save_model_path)
    print("Saved.")
    