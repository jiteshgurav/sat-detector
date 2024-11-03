from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn as nn

class FineTuneVisualTransformer:
    def __init__(self, model_selector, num_classes, learning_rate=1e-4, device=None):
        """
        Initializes the fine-tuning class for Visual Transformer models.

        Args:
            model_selector (VisualTransformerModels): Instance of VisualTransformerModels class.
            num_classes (int): Number of output classes for the classification task.
            learning_rate (float): Learning rate for fine-tuning.
            device (str): Device to use for training (e.g., "cuda" or "cpu").
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_selector = model_selector
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._prepare_model_for_finetuning()
        self.model.to(self.device)

    def _prepare_model_for_finetuning(self):
        """
        Modifies the Visual Transformer model for fine-tuning by adding a classification head.

        Returns:
            model: The modified Visual Transformer model with a classification head.
        """
        # Get the base model
        model = self.model_selector.model

        # Replace the last layer with a classification head
        if hasattr(model, "classifier"):  # Check if the model has a classifier
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        else:
            model.classifier = nn.Linear(model.config.hidden_size, self.num_classes)

        return model

    def train(self, train_dataset, val_dataset=None, epochs=5, batch_size=16):
        """
        Fine-tunes the model on the training dataset.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset (optional).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and evaluation.

        Returns:
            Trained model.
        """
        # Prepare data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

        # Set up the optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        num_training_steps = epochs * len(train_loader)
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            progress_bar = tqdm(train_loader, desc="Training")

            for batch in progress_bar:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix(loss=loss.item())

            # Validation step (optional)
            if val_loader:
                val_loss, val_accuracy = self.evaluate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        return self.model


    def evaluate(self, dataloader):
        """
        Evaluates the model on a given dataset.

        Args:
            dataloader: DataLoader for the dataset to evaluate on.

        Returns:
            (float, float): Tuple of validation loss and accuracy.
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Calculate accuracy
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_samples) * 100
        return avg_loss, accuracy
