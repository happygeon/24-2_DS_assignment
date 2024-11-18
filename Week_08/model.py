import torch
import torch.nn as nn


class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # First layer reduces dimensionality
            nn.BatchNorm1d(256),
            nn.ReLU(),            # Activation function
            nn.Dropout(0.2),      # Regularization
            nn.Linear(256, 128),  # Second layer
            nn.BatchNorm1d(128),
            nn.ReLU(),            # Activation function
            nn.Linear(128, 90)    # Final layer for 90 classes
        )

        

    def forward(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float()
        logits = self.classifier(image_features)
        return logits
    