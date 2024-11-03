# Visual Transformer Fine-Tuning and Evaluation for Satellite Images

This project provides a set of Python classes for loading, fine-tuning, and evaluating visual transformer models from Hugging Face on satellite imagery. The classes cover the entire workflow from model selection to fine-tuning and evaluation, and they include specific evaluation metrics commonly used in image classification and segmentation tasks, especially for satellite images.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Classes](#classes)
  - [1. VisualTransformerModels](#1-visualtransformermodels)
  - [2. FineTuneVisualTransformer](#2-finetunevisualtransformer)
  - [3. EvaluationMetrics](#3-Evaluation)
- [Usage Example](#usage-example)

## Overview

This project aims to provide a streamlined process for training and evaluating visual transformer models on satellite image data. Each class is designed to handle a specific part of the workflow:

1. **VisualTransformerModels**: Loads a pre-trained visual transformer model from Hugging Face.
2. **FineTuneVisualTransformer**: Fine-tunes the selected model on a specific classification task.
3. **EvaluationMetrics**: Computes classification metrics like accuracy, precision, recall, and F1-score.
4. **SatelliteImageEvaluationMetrics**: Computes metrics for satellite image segmentation tasks, such as Intersection over Union (IoU), Pixel Accuracy, and Dice Coefficient.

## Requirements

To use these classes, you’ll need the following packages:

```bash
pip install torch torchvision transformers scikit-learn
```

## Classes

### 1. VisualTransformerModels

This class is designed to load and configure a pre-trained visual transformer model from Hugging Face's model hub.

#### Methods

- **`__init__(model_name: str)`**: Initializes and loads the specified visual transformer model by name.
- **`load_model()`**: Loads the model and tokenizer from Hugging Face's model hub.

#### Usage

```python
from models import VisualTransformerModels

# Initialize a model
model_selector = VisualTransformerModels(model_name="vit")
```

### 2. FineTuneVisualTransformer

This class fine-tunes a selected visual transformer model on a classification task.

#### Parameters

- **`model_selector`** (`VisualTransformerModels`): Instance of `VisualTransformerModels` class.
- **`num_classes`** (`int`): Number of output classes for classification.
- **`learning_rate`** (`float`): Learning rate for fine-tuning.
- **`device`** (`str`): Device for training (e.g., "cuda" or "cpu").

#### Methods

- **`train(train_dataset, val_dataset=None, epochs=5, batch_size=16)`**: Fine-tunes the model on the provided training dataset.
- **`evaluate(dataloader)`**: Evaluates the model on a given dataset.

#### Usage

```python
from finetune import FineTuneVisualTransformer

fine_tuner = FineTuneVisualTransformer(model_selector=model_selector, num_classes=10, learning_rate=1e-4)
fine_tuned_model = fine_tuner.train(train_dataset=train_dataset, val_dataset=val_dataset, epochs=5, batch_size=16)
```


### 3. Evaluation

This class provides specialized evaluation metrics for satellite image segmentation tasks. It calculates metrics like IoU, Dice Coefficient, Pixel Accuracy, Precision, Recall, and F1-Score.

#### Parameters

- **`num_classes`** (`int`): Number of classes in the segmentation task.

#### Methods

- **`evaluate(true_labels, pred_labels)`**: Computes all metrics for segmentation evaluation.
- **`print_metrics(metrics)`**: Prints all metrics in a readable format.

#### Usage

```python
from evaluation import SatelliteImageEvaluationMetrics

# Initialize the evaluator
evaluator = SatelliteImageEvaluationMetrics(num_classes=5)
metrics = evaluator.evaluate(true_labels, pred_labels)
evaluator.print_metrics(metrics)
```

## Usage Example

Here’s a complete example of how you can use these classes to load, fine-tune, and evaluate a visual transformer on satellite images:

```python
# Step 1: Initialize a model selector
model_selector = VisualTransformerModels(model_name="vit")

# Step 2: Fine-tune the model
fine_tuner = FineTuneVisualTransformer(model_selector=model_selector, num_classes=5, learning_rate=1e-4)
fine_tuned_model = fine_tuner.train(train_dataset=train_dataset, val_dataset=val_dataset, epochs=5, batch_size=16)

# Step 3: Evaluate using general classification metrics (e.g., accuracy, precision)
evaluation_metrics = EvaluationMetrics(average='macro')
classification_metrics = evaluation_metrics.compute_metrics(predictions, labels)
evaluation_metrics.print_metrics(classification_metrics)

# Step 4: For segmentation tasks, evaluate using satellite-specific metrics
satellite_evaluator = SatelliteImageEvaluationMetrics(num_classes=5)
segmentation_metrics = satellite_evaluator.evaluate(true_labels, pred_labels)
satellite_evaluator.print_metrics(segmentation_metrics)
```

## Summary

This project provides a modular setup for training and evaluating visual transformer models on satellite images, with specific classes for model loading, fine-tuning, and evaluation. These classes can be extended and adapted to various computer vision tasks on satellite imagery data, providing a foundation for further exploration and experimentation in satellite image analysis.
