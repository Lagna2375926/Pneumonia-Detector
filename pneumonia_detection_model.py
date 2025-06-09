
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Model Configuration
class PneumoniaDetectionConfig:
    def __init__(self):
        self.IMAGE_SIZE = (224, 224)
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.LEARNING_RATE = 0.001
        self.NUM_CLASSES = 2
        self.CLASS_NAMES = ['Normal', 'Pneumonia']

# Data Preprocessing and Augmentation
def create_data_generators(train_dir, val_dir, test_dir, config):
    """Create data generators with augmentation for training and validation"""

    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )

    # Validation and test data (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

# Custom CNN Model
def create_custom_cnn(config):
    """Create a custom CNN model for pneumonia detection"""

    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*config.IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])

    return model

# Transfer Learning Model (DenseNet-121)
def create_transfer_learning_model(config):
    """Create a transfer learning model using DenseNet-121"""

    # Load pre-trained DenseNet-121
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMAGE_SIZE, 3)
    )

    # Freeze base model layers
    base_model.trainable = False

    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(config.NUM_CLASSES, activation='softmax')
    ])

    return model

# Model Training Function
def train_model(model, train_gen, val_gen, config, model_name):
    """Train the model with callbacks"""

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    return history

# Model Evaluation Functions
def evaluate_model(model, test_generator, config):
    """Comprehensive model evaluation"""

    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get true labels
    true_classes = test_generator.classes

    # Classification report
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=config.CLASS_NAMES,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    # ROC curve for binary classification
    if config.NUM_CLASSES == 2:
        fpr, tpr, _ = roc_curve(true_classes, predictions[:, 1])
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = None

    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'roc_auc': roc_auc
    }

# Visualization Functions
def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()

    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution function
def main():
    """Main function to run the pneumonia detection project"""

    # Initialize configuration
    config = PneumoniaDetectionConfig()

    # Data paths (update these with your actual data paths)
    train_dir = 'chest_xray/train'
    val_dir = 'chest_xray/val'
    test_dir = 'chest_xray/test'

    print("=== Chest X-Ray Pneumonia Detection Project ===")
    print(f"Image Size: {config.IMAGE_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Number of Classes: {config.NUM_CLASSES}")
    print(f"Class Names: {config.CLASS_NAMES}")

    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir, val_dir, test_dir, config
    )

    # Model 1: Custom CNN
    print("\n=== Training Custom CNN Model ===")
    custom_model = create_custom_cnn(config)
    custom_model.summary()

    # Model 2: Transfer Learning (DenseNet-121)
    print("\n=== Training Transfer Learning Model (DenseNet-121) ===")
    transfer_model = create_transfer_learning_model(config)
    transfer_model.summary()

    print("\nModels created successfully!")
    print("\nTo train the models, run:")
    print("history1 = train_model(custom_model, train_gen, val_gen, config, 'custom_cnn')")
    print("history2 = train_model(transfer_model, train_gen, val_gen, config, 'densenet_transfer')")

if __name__ == "__main__":
    main()
