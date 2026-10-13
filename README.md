# CIFAR-10 Image Classifier

This project trains a convolutional neural network (CNN) to recognize 10 kinds of small color images, such as cars, birds, and dogs. It uses TensorFlow/Keras and the CIFAR-10 dataset, which contains 50,000 training images and 10,000 test images.

## How the model improved

We developed the model in stages. Each stage adds changes intended to help the model learn image features and perform better on images it has not seen before.

| Level | Model changes | Recorded test accuracy |
| --- | --- | --- |
| 1. Initial baseline | A simple CNN with three convolutional blocks, trained for 10 epochs. The project notes report approximately **75–80%**; an exact test score for this stage is not recorded. | About **75–80%** |
| 2. Deeper CNN | Added a fourth convolutional block, batch normalization, a wider 256-unit dense layer, data augmentation, and longer training. | **86.01%** |
| 3. Dropout tuning | Adjusted dropout rates to reduce regularization while keeping the deeper network and training callbacks. | **86.51%** |

The latest recorded test score is **0.50 percentage points higher** than the 86.01% score before dropout tuning. The notebook's best-checkpoint evaluation records 86.51%. The target of 90% is a goal, not a result demonstrated so far.

These are successive model iterations, not a strict scientific ablation study: multiple training and architecture changes were made between some stages, so the accuracy difference cannot be attributed to one change alone. The Level 1 score is approximate, and the notebook does not contain an exact test result for that initial run.

## What the final model uses

- Four convolutional blocks to learn visual patterns at increasing levels of detail.
- Batch normalization and dropout to support stable training and reduce overfitting.
- Random horizontal flips, rotations, and zooms to create varied training examples.
- Pixel normalization, scaling image values from 0–255 to 0–1.
- Early stopping, learning-rate reduction, and checkpointing to retain a strong validation-loss model.

## Run the notebook

Install Python, then install the required packages:

```bash
python -m pip install tensorflow notebook
```

From this project folder, start Jupyter:

```bash
jupyter notebook
```

Open `23_A_Image_Classification_using_CIFAR_10_dataset_04Oct.ipynb` and run its cells in order. Training can take a while and is configured for up to 50 epochs. The notebook downloads CIFAR-10 through TensorFlow if the dataset is not already available.

## Load the saved model

The project includes `best_cifar10_model.keras`. Load it with:

```python
from tensorflow.keras.models import load_model

model = load_model("best_cifar10_model.keras")
```

## Project files

- `23_A_Image_Classification_using_CIFAR_10_dataset_04Oct.ipynb` - training and evaluation notebook.
- `best_cifar10_model.keras` - saved Keras model.
- `Model_Summary.md` - detailed notes on the architecture and experiments.
- `Model_Improvement_Report.html` - model-improvement report.
