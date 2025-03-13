import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df, test_size=0.2, val_size=0.125, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    Args:
        df (pd.DataFrame): Input DataFrame containing filenames and labels.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        random_state (int): Random seed for reproducibility.
    Returns:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        test_df (pd.DataFrame): Test set.
    """
    # Map species IDs to string labels
    species_map = {1: 'Cat', 2: 'Dog'}
    df['species_name'] = df['species'].map(species_map)
    df['class_id'] = df['class_id'] - 1  # ✅ Shift labels from [1, 37] → [0, 36]

    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['class_id'], random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['class_id'], random_state=random_state)
    return train_df, val_df, test_df