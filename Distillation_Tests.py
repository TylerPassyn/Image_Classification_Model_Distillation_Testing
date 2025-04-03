from PIL import Image
from transformers import ViTFeatureExtractor
from transformers import ViTConfig, ViTForImageClassification
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTImageProcessor

from Image_Distiller import Distill_Model_VIT_to_VIT as Distill_VIT

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import os
import sys



def main():

    logits = pd.read_csv('/deac/csc/classes/csc373/passta23/model_distillation/Results/Main_Model_Outputs/logits_output.csv')

    train_data, test_data = train_test_split(logits, test_size=0.2, random_state=42)

    logits_train = train_data["Logits"].values
    logits_test = test_data["Logits"].values
    labels_train = train_data["Image Class"].values
    labels_test = test_data["Image Class"].values

    #the number of classes equals the number of logits
    num_classes = len(logits_train[0].split(",")) 
    print("Num_classes " + str(num_classes))

    logits_train = np.array([np.fromstring(logit[1:-1], sep=',') for logit in logits_train])
    logits_test = np.array([np.fromstring(logit[1:-1], sep=',') for logit in logits_test])
    print("Train_logits " + str(logits_train.shape))
    print("Test_logits " + str(logits_test.shape))

    # Convert logits to PyTorch tensors
    logits_train = torch.tensor(logits_train, dtype=torch.float32)
    logits_test = torch.tensor(logits_test, dtype=torch.float32)

    # Convert labels to PyTorch tensors
    #labels_train = torch.tensor(labels_train, dtype=torch.long)
    #labels_test = torch.tensor(labels_test, dtype=torch.long)
    #logits, extracted_features, num_hidden_layers, num_classes, hidden_size=768, attention_heads=12
    # Create the model

    #extracted featues /deac/csc/classes/csc373/passta23/model_distillation/Results/Main_Model_Outputs/features_batch_31.npy
    extracted_features = np.load('/deac/csc/classes/csc373/passta23/model_distillation/Results/Main_Model_Outputs/all_features.npy')
    extracted_features = torch.tensor(extracted_features, dtype=torch.float32)



    distiller = Distill_VIT(
        num_hidden_layers=12,
        num_classes=num_classes,
        hidden_size = 512,
        attention_heads=8
    )

    #distilled_model = distiller.distill(logits_train, extracted_features, num_epochs=5, learning_rate=0.0001)

    # Save the distilled model
    #distiller.save_model("distilled_model")

    model_path = '/deac/csc/classes/csc373/passta23/model_distillation/distilled_model'

    model = ViTForImageClassification.from_pretrained(model_path, from_tf=False, config=model_path)

    #get model size in layers
    model_size = model.num_parameters()
    print("Model size: ", model_size)
    #get model size in MB
    model_size = model_size * 1e-6
    print("Model size in MB: ", model_size)
    #predict using the row nums of the test data
    logits_test = logits_test[0:1]
    labels_test = labels_test[0:1]



if __name__ == "__main__":
    main()

