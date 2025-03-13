from preprocess import preprocess_data
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(train_generator, val_generator):
    """
    Train a model using the provided data generators.
    Args:
        train_generator: Data generator for the training set.
        val_generator: Data generator for the validation set.
    Returns:
        model: Trained model.
    """
    # Load the ResNet50 model without the top classification layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    base_model.trainable = False
    #for layer in base_model.layers[:-10]:  # Freeze all but last 10 layers
    #    layer.trainable = False
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(37, activation='softmax')(x) # 37 breeds 

    # Define the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.00001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint('best_model.h5',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   restore_best_weights=True,
                                   verbose=1)


    #print("Number of classes in train_generator:", len(train_generator.class_indices))
    #print("Class indices:", train_generator.class_indices)
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=50,
        callbacks=[checkpoint, early_stopping]
    )

    return model

if __name__ == '__main__':
    # Define paths
    dataset_path = '../data/'  # Update with the correct path
    output_dir = '../data/preprocessed'  # Update with the correct path

    # Preprocess the data and get the data generators
    train_generator, val_generator, test_generator = preprocess_data(dataset_path, output_dir)

    # Train the model
    model = train_model(train_generator, val_generator)

    # Save the final model
    model.save('pet_classification_model.h5')