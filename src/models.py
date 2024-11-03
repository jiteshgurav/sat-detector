from transformers import ViTModel, DeiTModel, BeitModel, AutoFeatureExtractor
import torch

class VisualTransformerModels:
    def __init__(self, model_name="vit"):
        """
        Initializes a Visual Transformer model based on the given model name.

        Args:
            model_name (str): The name of the model to load.
                              Supported values are 'vit', 'deit', 'beit'.
        """
        self.model_name = model_name.lower()
        self.model, self.feature_extractor = self._load_model_and_extractor(self.model_name)
    
    def _load_model_and_extractor(self, model_name):
        """
        Loads the specified Visual Transformer model and its feature extractor.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            model: The pre-trained Visual Transformer model.
            feature_extractor: The feature extractor to preprocess input images.
        """
        if model_name == "vit":
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        elif model_name == "deit":
            model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
            feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
        elif model_name == "beit":
            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
            feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
        else:
            raise ValueError("Model name not recognized. Use 'vit', 'deit', or 'beit'.")
        return model, feature_extractor

    def preprocess(self, images):
        """
        Preprocesses images for the model.

        Args:
            images (list of PIL.Image or numpy arrays): List of images to preprocess.

        Returns:
            torch.Tensor: Preprocessed image tensors.
        """
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        return inputs

    def forward(self, images):
        """
        Passes the images through the model.

        Args:
            images (list of PIL.Image or numpy arrays): List of images to pass through the model.

        Returns:
            torch.Tensor: Model output.
        """
        inputs = self.preprocess(images)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
