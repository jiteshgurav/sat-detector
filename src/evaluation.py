from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch

class SatelliteImageEvaluationMetrics:
    def __init__(self, num_classes):
        """
        Initializes the evaluation metrics class for satellite image segmentation.

        Args:
            num_classes (int): Number of classes in the segmentation task.
        """
        self.num_classes = num_classes

    def _compute_confusion_matrix(self, true_labels, pred_labels):
        """
        Computes the confusion matrix for multi-class segmentation.

        Args:
            true_labels (numpy.ndarray): Ground truth labels.
            pred_labels (numpy.ndarray): Predicted labels.

        Returns:
            numpy.ndarray: Confusion matrix.
        """
        return confusion_matrix(true_labels, pred_labels, labels=list(range(self.num_classes)))

    def compute_iou(self, confusion_matrix):
        """
        Calculates the Intersection over Union (IoU) for each class.

        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix.

        Returns:
            dict: Dictionary of IoU for each class and mean IoU.
        """
        intersection = np.diag(confusion_matrix)
        union = (
            np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - intersection
        )
        iou = intersection / np.maximum(union, 1)  # Avoid division by zero
        mean_iou = np.mean(iou)
        
        return {"iou": iou, "mean_iou": mean_iou}

    def compute_pixel_accuracy(self, confusion_matrix):
        """
        Calculates pixel accuracy.

        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix.

        Returns:
            float: Pixel accuracy.
        """
        return np.diag(confusion_matrix).sum() / confusion_matrix.sum()

    def compute_dice_coefficient(self, confusion_matrix):
        """
        Calculates Dice coefficient for each class.

        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix.

        Returns:
            dict: Dice coefficient for each class and mean Dice coefficient.
        """
        intersection = np.diag(confusion_matrix)
        sums = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0)
        dice = 2 * intersection / np.maximum(sums, 1)  # Avoid division by zero
        mean_dice = np.mean(dice)
        
        return {"dice": dice, "mean_dice": mean_dice}

    def compute_precision_recall_f1(self, true_labels, pred_labels):
        """
        Calculates precision, recall, and F1-score for each class.

        Args:
            true_labels (numpy.ndarray): Ground truth labels.
            pred_labels (numpy.ndarray): Predicted labels.

        Returns:
            dict: Precision, recall, and F1-score for each class.
        """
        precision = precision_score(true_labels, pred_labels, average=None, labels=list(range(self.num_classes)))
        recall = recall_score(true_labels, pred_labels, average=None, labels=list(range(self.num_classes)))
        f1 = f1_score(true_labels, pred_labels, average=None, labels=list(range(self.num_classes)))
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def evaluate(self, true_labels, pred_labels):
        """
        Calculates all metrics for segmentation evaluation.

        Args:
            true_labels (torch.Tensor): Ground truth label tensor.
            pred_labels (torch.Tensor): Predicted label tensor.

        Returns:
            dict: Dictionary containing all evaluation metrics.
        """
        # Flatten tensors to compute metrics
        true_labels = true_labels.view(-1).cpu().numpy()
        pred_labels = pred_labels.view(-1).cpu().numpy()

        # Confusion matrix
        conf_matrix = self._compute_confusion_matrix(true_labels, pred_labels)

        # Calculate metrics
        iou = self.compute_iou(conf_matrix)
        pixel_accuracy = self.compute_pixel_accuracy(conf_matrix)
        dice = self.compute_dice_coefficient(conf_matrix)
        prf = self.compute_precision_recall_f1(true_labels, pred_labels)

        # Aggregate metrics into a single dictionary
        metrics = {
            "IoU": iou,
            "Pixel Accuracy": pixel_accuracy,
            "Dice Coefficient": dice,
            "Precision": prf["precision"],
            "Recall": prf["recall"],
            "F1 Score": prf["f1_score"]
        }
        return metrics

    def print_metrics(self, metrics):
        """
        Prints the metrics in a readable format.

        Args:
            metrics (dict): Dictionary of evaluation metrics.
        """
        print("Evaluation Metrics for Satellite Image Segmentation:")
        print(f"Mean IoU: {metrics['IoU']['mean_iou']:.4f}")
        print(f"Pixel Accuracy: {metrics['Pixel Accuracy']:.4f}")
        print(f"Mean Dice Coefficient: {metrics['Dice Coefficient']['mean_dice']:.4f}")
        print("Class-wise Metrics:")
        for cls in range(self.num_classes):
            print(f"Class {cls}:")
            print(f"  IoU: {metrics['IoU']['iou'][cls]:.4f}")
            print(f"  Dice: {metrics['Dice Coefficient']['dice'][cls]:.4f}")
            print(f"  Precision: {metrics['Precision'][cls]:.4f}")
            print(f"  Recall: {metrics['Recall'][cls]:.4f}")
            print(f"  F1 Score: {metrics['F1 Score'][cls]:.4f}")
