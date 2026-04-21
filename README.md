This project builds a deep learning pipeline for **multi-label classification** on the **NIH Chest X-ray dataset** using **PyTorch** and a **pretrained DenseNet121** backbone.

The main goal was to create a clean and practical training pipeline that covers:

- dataset preparation - Creating **DataPipeline**
- preprocessing and augmentation
- train/validation/test splitting
- **DataLoader Pipeline setup**
- class imbalance handling
- DenseNet121 fine-tuning for multi-label prediction

---

## 1. Project Overview

The NIH Chest X-ray dataset contains chest X-ray images where a single image can have **multiple disease findings at the same time**.

Because of that, this is a **multi-label classification problem**, not a single-label classification problem.

That design decision affected several parts of the pipeline:

- labels must be stored as **multi-hot vectors**
- the model output layer must produce one score per class
- the loss function must be **`BCEWithLogitsLoss`**
- prediction logic must use **`sigmoid` + threshold**
- class imbalance must be handled per class using **`pos_weight`**
- Increasing training efficiency using Torchvision v2, Nvidia DALI to shift image transforms and augmentation operations to GPU to avoid CPU bottle necking
- NVIDIA DALI improved image transform, augmentation and normalization speed by upto 55% and improved training speed.

---

## 2. Dataset Preparation

A custom PyTorch dataset class was used to load the NIH data.

### Dataset Pipeline

The custom `NIHDataset` class was designed to:

- read the NIH metadata CSV
- parse the `Finding Labels` column
- build a list of all unique classes
- create label-to-index mappings
- scan image folders and map image names to full file paths
- load each image from disk
- return an image and its corresponding target

### Key internal attributes

The dataset stores useful attributes such as:

- `all_labels` → list of all unique disease labels
- `class_to_idx` → mapping from class name to integer index
- `idx_to_class` → reverse mapping
- `image_paths` → mapping from image file name to its absolute path

This allowed us to later determine:

- number of classes
- class names
- class imbalance statistics

---

## 3. Label Processing

The NIH data is not a regular single-class dataset.

Each image may contain findings like:

- Atelectasis
- Infiltration
- Effusion
- Mass
- Pneumonia

and possibly more than one of them at once.

So instead of returning one integer class index, the dataset should return a **multi-hot encoded target vector**.

### Example

If we had 5 classes:


["Atelectasis", "Cardiomegaly", "Effusion", "Mass", "Nodule"]

Results:
<img width="869" height="734" alt="image" src="https://github.com/user-attachments/assets/17f4c0c1-9933-4a24-a253-b0190a339f67" />
<img width="731" height="475" alt="image" src="https://github.com/user-attachments/assets/2de25786-0b0d-4a21-abeb-c6a8c7761196" />





###Model Comparison: Filter out models with accuracy on 5 Epochs.

```python
Comparison Between RestNet and MobileNet:
Model: resnet18
Train losses: [1.0415923982591482, 0.9263832987374956, 0.8471168558968697, 0.7686603944672737, 0.6759863143396649]
Validation accuracy: [71.58441352519127, 71.4734213342847, 76.83156935029928, 78.69227415071154, 79.95481032227374]
Test accuracy: 79.92%
----------------------------------------
Model: mobilenet_v2
Train losses: [1.0668273168360205, 0.9357404465974591, 0.8636343249870493, 0.8025839413290793, 0.7568629685707372]
Validation accuracy: [70.04915368454434, 74.85194434534428, 76.65913505371229, 78.03107781345385, 77.60375787846355]
Test accuracy: 77.65%

Model: vgg16
Train losses: [1.2481776367092676, 1.1596507639931584, 1.10912646951963, 1.0740565112575633, 1.0278751656789553]
Validation accuracy: [53.75074325127839, 65.11079399056567, 70.6267094779403, 69.7883220359139, 75.65386292464423]
Test accuracy: 75.54%
----------------------------------------
Model: densenet121
Train losses: [1.0317207735150251, 0.9203205055693057, 0.8477077343741761, 0.7925363405020915, 0.7338018331376148]
Validation accuracy: [73.37812661037776, 74.20184722717723, 75.41760811828595, 76.06572323304395, 78.84330281048084]
Test accuracy: 78.76%
----------------------------------------


###Final Model Selection: DenseNet121:

Torch CPU threads: 1
DALI threads: 4
Dataset size: 112120
Num classes: 15
Train samples: 78484
Val samples: 16818
Test samples: 16818
[Training] Epoch 1:
        Step 818/818 - Loss: 0.939 | Acc: 72.90% | Time Elapsed: 5.92Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 1:
        Validation Accuracy: 75.68%
[Training] Epoch 2:
        Step 818/818 - Loss: 0.925 | Acc: 74.36% | Time Elapsed: 5.84Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 2:
        Validation Accuracy: 77.09%
[Training] Epoch 3:
        Step 818/818 - Loss: 0.835 | Acc: 77.72% | Time Elapsed: 10.22Mins | Avg Time Per Batch: 0.75Secs
[Validation] Epoch 3:
        Validation Accuracy: 76.13%
[Training] Epoch 4:
        Step 818/818 - Loss: 0.800 | Acc: 78.29% | Time Elapsed: 5.87Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 4:
        Validation Accuracy: 80.29%
[Training] Epoch 5:
        Step 818/818 - Loss: 0.713 | Acc: 79.90% | Time Elapsed: 5.81Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 5:
        Validation Accuracy: 79.23%
[Training] Epoch 6:
        Step 818/818 - Loss: 0.668 | Acc: 81.35% | Time Elapsed: 5.79Mins | Avg Time Per Batch: 0.42Secs
[Validation] Epoch 6:
        Validation Accuracy: 81.55%
[Training] Epoch 7:
        Step 818/818 - Loss: 0.576 | Acc: 83.63% | Time Elapsed: 5.77Mins | Avg Time Per Batch: 0.42Secs
[Validation] Epoch 7:
        Validation Accuracy: 82.36%
[Training] Epoch 8:
        Step 818/818 - Loss: 0.552 | Acc: 85.16% | Time Elapsed: 5.89Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 8:
        Validation Accuracy: 84.75%
[Training] Epoch 9:
        Step 818/818 - Loss: 0.461 | Acc: 87.22% | Time Elapsed: 5.80Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 9:
        Validation Accuracy: 86.83%
[Training] Epoch 10:
        Step 818/818 - Loss: 0.423 | Acc: 87.85% | Time Elapsed: 5.80Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 10:
        Validation Accuracy: 85.53%

Final summary
Train losses: [1.0365152495997458, 0.9103261829879873, 0.8344724423873687, 0.7758005154307722, 0.7038509688342405, 0.6541387189571315, 0.5674298937571369, 0.5000766215158267, 0.4498674208639886, 0.40396104330392807]
Validation accuracy: [75.67548500881834, 77.08563590045071, 76.12972761120909, 80.28728199098569, 79.22908093278464, 81.54732510288066, 82.36057221242406, 84.75485008818342, 86.83127572016461, 85.52968841857731]

[Test evaluation]
        Test Accuracy: 85.11%

```
#Final Results with DenseNet121
Results:
<img width="869" height="734" alt="image" src="https://github.com/user-attachments/assets/17f4c0c1-9933-4a24-a253-b0190a339f67" />
<img width="731" height="475" alt="image" src="https://github.com/user-attachments/assets/2de25786-0b0d-4a21-abeb-c6a8c7761196" />






