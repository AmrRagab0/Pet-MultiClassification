
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_augmentation_generator():
    """
    Create an ImageDataGenerator for data augmentation.
    Returns:
        train_datagen (ImageDataGenerator): Data generator for training set.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return train_datagen