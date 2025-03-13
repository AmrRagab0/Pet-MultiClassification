import os
import pandas as pd
from resize_images import resize_images_in_directory
from normalize import normalize_image
from split_data import split_dataset
from Data_augmentation import create_augmentation_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(dataset_path, output_dir, target_size=(224, 224)):
    """
    Preprocess the dataset: resize, normalize, split, and augment.
    Args:
        dataset_path (str): Path to the raw dataset.
        output_dir (str): Directory to save preprocessed data.
        target_size (tuple): Target size for resizing images.
    """

   
    # Load metadata
    metadata_path = os.path.join(dataset_path, 'annotations/list.txt')
    df = pd.read_csv(metadata_path, sep=' ', header=None, names=['filename', 'class_id', 'species', 'breed_id'])

    # Resize images
    resized_dir = os.path.join(output_dir, 'resized')
    resize_images_in_directory(os.path.join(dataset_path, 'images'), resized_dir, target_size)

    # Normalize images (optional, can be done during training)
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)

    # Save split data
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)


    train_df['filename'] = train_df['filename'] + '.jpg'
    val_df['filename'] = val_df['filename'] + '.jpg'
    test_df['filename'] = test_df['filename'] + '.jpg'
    # Create data generators
    train_datagen = create_augmentation_generator()
    # (Add code to create data generators for training, validation, and test sets)
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only rescale for validation and test sets

    # Create data generators for training, validation, and test sets
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=resized_dir,
        x_col='filename',
        y_col='species_name',  # Use 'species_name' for species classification or 'breed_name' for breed classification
        target_size=target_size,
        batch_size=32,
        class_mode='categorical'
    )
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=resized_dir,
        x_col='filename',
        y_col='species_name',  # Use 'species_name' for species classification or 'breed_name' for breed classification
        target_size=target_size,
        batch_size=32,
        class_mode='categorical'
    )
    test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=resized_dir,
    x_col='filename',
    y_col='species_name',  # Use 'species_name' for species classification or 'breed_name' for breed classification
    target_size=target_size,
    batch_size=32,
    class_mode='categorical'
    )
    print("Class indices:", train_generator.class_indices)

    # Return the data generators
    return train_generator, val_generator, test_generator

if __name__ == '__main__':
    dataset_path = '../data/'
    output_dir = '../data/preprocessed'
    train_generator, val_generator, test_generator = preprocess_data(dataset_path, output_dir)
