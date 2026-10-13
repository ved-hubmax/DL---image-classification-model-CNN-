# CIFAR-10 Model Summary

## Overview
This project builds and improves a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The dataset contains 60,000 color images across 10 classes, with 50,000 used for training and 10,000 used for testing. The final model was designed to improve both accuracy and generalization performance on this dataset.

## Initial Model Performance
The initial CNN model was a good starting point, but it was relatively shallow. It used a basic 3-block architecture and trained for only 10 epochs. While it achieved a decent starting accuracy, it plateaued around 75–80% because the model lacked enough depth and training time to learn the complex patterns in CIFAR-10 images.

## Latest Recorded Baseline
The notebook's recorded run of the improved CNN reached **86.01% test accuracy** after 50 epochs. Its final logged validation accuracy was **86.28%**. These are baseline results from before the tuning changes below; 90% test accuracy has not yet been demonstrated.

## Improvements Made
To improve the model performance, the following changes were made:

### 1. Increased the Depth of the CNN
A 4th convolutional block was added to the model with 256 filters. This increased the network's ability to learn more detailed and complex image features.

```python
# Block 4
model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Dropout(0.4))
```

### 2. Increased Dense Layer Capacity
The dense layer was expanded from 128 neurons to 256 neurons. This gave the classifier more capacity to interpret the learned features before the final output layer.

```python
model.add(layers.Dense(256, activation='relu'))
```

### 3. Tuned Dropout Regularization
The previous dropout rates were 0.2, 0.3, 0.4, and 0.5 across the convolution blocks, plus 0.5 after the dense layer. For the next run, these are reduced to 0.15, 0.25, 0.35, and 0.4, with 0.4 after the dense layer. This tuning attempt reduces regularization so the model can fit more of the training data; its effect must be checked using the next run's validation and test results.

### 4. Used Data Augmentation
Data augmentation was applied using random flipping, rotation, and zoom. This helped the model learn more robust features and prevented overfitting.

```python
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])
```

### 5. Normalized Input Data
The image pixel values were scaled from [0, 255] to [0, 1] before training. This made optimization smoother and improved convergence.

```python
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
```

### 6. Increased Training Time
The tuned run allows up to 50 epochs. EarlyStopping can stop training earlier if validation loss does not improve for 15 epochs.

```python
history = model.fit(train_images, train_labels,
                    epochs=50,
                    validation_split=0.1,
                    callbacks=[model_checkpoint, early_stopping, reduce_lr])
```

### 7. Tuned and Added Callbacks
The model included:
- EarlyStopping now waits 15 epochs without validation-loss improvement and restores the best in-memory weights.
- ReduceLROnPlateau lowers the learning rate by a factor of 0.5 after 5 epochs without validation-loss improvement.
- ModelCheckpoint saves the model with the best validation loss to `best_cifar10_model.keras`. The notebook reloads this checkpoint before evaluating the test set.

These callbacks are intended to use training time more effectively and preserve the best validation-loss model.

## Tuning Goal and Result
The next run targets **90% test accuracy**, but that result is not guaranteed. The recorded baseline is 86.01% test accuracy. Run the updated training and evaluation cells, then replace this section with the measured test accuracy; do not treat training or validation accuracy as test accuracy.

## Conclusion
The improved CNN adds depth, batch normalization, data augmentation, and dropout. The latest tuning changes are a measured experiment; their impact on test accuracy remains unverified until the notebook is run again.
