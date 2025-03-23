import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

def get_data_generators(train_dir, test_dir):
    
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=10,      
        width_shift_range=0.1, 
        height_shift_range=0.1  
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False 
    )
    return train_generator, test_generator
