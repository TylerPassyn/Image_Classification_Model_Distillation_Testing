from PIL import Image
from transformers import ViTFeatureExtractor
from transformers import ViTConfig, ViTForImageClassification
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor
from tqdm import tqdm


class Distill_Model_VIT_to_VIT: 
    def __init__(self, num_hidden_layers, num_classes, hidden_size=768, attention_heads=12):
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.model = self.init_model()

       
    def init_model(self):
        # Define a smaller ViT student model (fewer layers, smaller hidden size)
        student_config = ViTConfig(
            num_hidden_layers=self.num_hidden_layers,
                hidden_size=self.hidden_size,       
                num_attention_heads=self.attention_heads, 
                num_labels=self.num_classes
            )
        student_model = ViTForImageClassification(student_config)


        return student_model
    
    
    def calculate_kl_loss(self, teacher_logits, student_logits, temperature=4.0):
        teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
        kl_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False) * (temperature ** 2)
        return kl_loss
    
   
    def distill(self, logits, extracted_features, num_epochs=10, learning_rate=0.001):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)


        #get extracted features from the teacher model
        #run it through the student model
        #calculate KL divergence loss
        #update student model parameters using backpropagation
        #repeat for all features and logits for num_epochs
        for epoch in range(num_epochs):
            total_loss = 0.0
            self.model.train()

            
            with tqdm(total=len(logits), desc=f"Epoch {epoch+1}/{num_epochs}", unit="feature") as pbar:
                for feature, logit in zip(extracted_features, logits):
                        feature = feature.to(device)
                        logit = logit.to(device)
                       
                        optimizer.zero_grad()  # Reset gradients

                        # Forward pass through student model
                        student_output = self.model(feature.unsqueeze(0))  # Add batch dimension
                        student_logits = student_output.logits  # Extract the logits

                        # Compute KL divergence loss
                        loss = self.calculate_kl_loss(logit, student_logits)

                        # Backpropagate
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        pbar.update(1)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(extracted_features):.4f}")
        print("Distillation complete.")
           
                
        return self.model
    
    def save_model(self, path):
        # Save the distilled model
        self.model.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        # Load the distilled model
        self.model = ViTForImageClassification.from_pretrained(path)
        print(f"Model loaded from {path}")
        return self.model
    
    def predict(self, image_path):
        # Load and preprocess the image
        feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image = Image.open(image_path)
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Forward pass through the distilled model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get the predicted class
        predicted_class = logits.argmax(dim=1).item()
        return predicted_class
    





