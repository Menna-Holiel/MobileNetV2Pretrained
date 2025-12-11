````markdown
# Face Recognition with MobileNetV2 (Transfer Learning)

## Overview

This project implements a **face recognition system** using **MobileNetV2** with transfer learning.  
The model is trained on a **balanced subset of VGGFace2**, consisting of **15 classes** with **900 images per class**.  

The goal is to achieve **high classification accuracy** while leveraging pretrained ImageNet features and advanced data augmentation.

---

## Features

- **Transfer Learning:** MobileNetV2 backbone pretrained on ImageNet.
- **Data Augmentation & Normalization:**
  - Random horizontal flip
  - Random rotation
  - Random zoom
  - Random contrast
  - Random brightness
  - Rescaling to [0,1]
- **Training Strategy:**
  - Frozen backbone training
  - Fine-tuning last 30 layers
  - BatchNormalization layers frozen during fine-tuning
- **Callbacks:**
  - `ModelCheckpoint` – save best model weights
  - `EarlyStopping` – prevent overfitting
  - `ReduceLROnPlateau` – reduce learning rate when stuck
- **Evaluation:**
  - Test accuracy & loss
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
  - ROC curves & AUC per class
  - Training & fine-tuning accuracy/loss curves

---

## Dataset

- Path: `VGGFace2_balanced_900_albumentations`
- Classes: 15
- Images per class: 900 (balanced)
- Image size: 224x224
- Split: Train 70% / Validation 15% / Test 15%

---

## Installation

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install tensorflow matplotlib seaborn scikit-learn
````

---

## Usage

1. Update the dataset path in the notebook or script:

```python
BALANCED_DATASET_PATH = "VGGFace2_balanced_900_albumentations"
```

2. **Stage 1:** Train MobileNetV2 with frozen backbone (optional if weights exist)

```python
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[checkpoint, early_stop, reduce_lr])
```

3. **Stage 2:** Fine-tune last layers with low learning rate

```python
history_fine = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[checkpoint_fine, reduce_lr, early_stop_fine])
```

4. Evaluate the model:

```python
test_loss, test_acc = model.evaluate(test_ds)
```

5. Generate metrics & plots:

   * Accuracy/Loss curves
   * Confusion Matrix
   * Classification Report
   * ROC curves & AUC

---

## Results

* **Validation Accuracy:** >90%
* **Test Accuracy:** >90%
* Detailed evaluation plots and metrics per class included.

---

## Notes

* BatchNormalization layers are frozen during fine-tuning to **stabilize training**.
* Efficient **tf.data pipeline** is used for preprocessing, augmentation, and batching.
* Model weights are automatically saved with `ModelCheckpoint`.

---




