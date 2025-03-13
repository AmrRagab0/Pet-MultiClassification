import os
import pandas as pd
from resize_images import resize_images_in_directory
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
    Returns:
        train_generator: Data generator for the training set.
        val_generator: Data generator for the validation set.
        test_generator: Data generator for the test set.
    """
    # Load metadata
    metadata_path = os.path.join(dataset_path, 'annotations/list.txt')
    df = pd.read_csv(metadata_path, sep=' ', header=None, names=['filename', 'class_id', 'species', 'breed_id'])

    # Add .jpg extension to filenames
    df['filename'] = df['filename'] + '.jpg'

    # Define breed mapping for cats and dogs
    cat_breeds = {
        1: 'Abyssinian', 2: 'Bengal', 3: 'Birman', 4: 'Bombay', 5: 'British_Shorthair',
        6: 'Egyptian_Mau', 7: 'Maine_Coon', 8: 'Persian', 9: 'Ragdoll', 10: 'Russian_Blue',
        11: 'Siamese', 12: 'Sphynx'  # Cat breeds
    }

    dog_breeds = {
        1: 'american_bulldog', 2: 'american_pit_bull_terrier', 3: 'basset_hound',
        4: 'beagle', 5: 'boxer', 6: 'chihuahua', 7: 'english_cocker_spaniel',
        8: 'english_setter', 9: 'german_shorthaired', 10: 'great_pyrenees',
        11: 'havanese', 12: 'japanese_chin', 13: 'keeshond', 14: 'leonberger',
        15: 'miniature_pinscher', 16: 'newfoundland', 17: 'pomeranian', 18: 'pug',
        19: 'saint_bernard', 20: 'samoyed', 21: 'scottish_terrier', 22: 'shiba_inu',
        23: 'staffordshire_bull_terrier', 24: 'wheaten_terrier', 25: 'yorkshire_terrier'  # Dog breeds
    }

    # Combine cat and dog breeds into a single breed_map
    breed_map = {}
    for breed_id, breed_name in cat_breeds.items():
        breed_map[breed_id] = breed_name  # Cat breeds have IDs 1-12
    for breed_id, breed_name in dog_breeds.items():
        breed_map[breed_id + 12] = breed_name  # Dog breeds have IDs 13-37


    # Add breed names to the DataFrame
    df['breed_name'] = df.apply(lambda row: breed_map[row['breed_id']] if row['species'] == 1 else breed_map[row['breed_id'] + 12], axis=1)
    
    all_breeds = list(breed_map.values())
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)

    # Save split data
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # Resize images
    resized_dir = os.path.join(output_dir, 'resized')
    resize_images_in_directory(os.path.join(dataset_path, 'images'), resized_dir, target_size)

    # Create data generators
    train_datagen = create_augmentation_generator()
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Only rescale for validation and test sets
    print("Number of unique breeds:", df['breed_name'].nunique())
    # Create data generators for training, validation, and test sets
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=resized_dir,
        x_col='filename',
        y_col='breed_name',  # Use 'breed_name' for breed classification
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        classes=all_breeds 
    )

    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=resized_dir,
        x_col='filename',
        y_col='breed_name',  # Use 'breed_name' for breed classification
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        classes=all_breeds
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=resized_dir,
        x_col='filename',
        y_col='breed_name',  # Use 'breed_name' for breed classification
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        classes=all_breeds
    )

    # Print class indices
    print("Class indices:", train_generator.class_indices)

    # Return the data generators
    return train_generator, val_generator, test_generator

if __name__ == '__main__':
    dataset_path = '../data/'
    output_dir = '../data/preprocessed'
    train_generator, val_generator, test_generator = preprocess_data(dataset_path, output_dir)