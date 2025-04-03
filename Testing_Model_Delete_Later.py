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

#open saved model
#/deac/csc/classes/csc373/passta23/model_distillation/distilled_model/config.json
#/deac/csc/classes/csc373/passta23/model_distillation/distilled_model/model.safetensors

model_path = '/deac/csc/classes/csc373/passta23/model_distillation/distilled_model'

model = ViTForImageClassification.from_pretrained(model_path, from_tf=False, config=model_path)

print("Model loaded")



