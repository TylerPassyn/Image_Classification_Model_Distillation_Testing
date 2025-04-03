
''''
@misc{wu2020visual,
      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, 
      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},
      year={2020},
      eprint={2006.03677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''

#https://huggingface.co/google/vit-large-patch16-224


from PIL import Image
import requests
from transformers import ViTFeatureExtractor, ViTForImageClassification
import time

#load 1000 images from ImageNet


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)



feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')

model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')



start_time = time.time()
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
end_time = time.time()

model_size = model.num_parameters()
print("Model size (in parameters):", model_size)
print("Model size (in MB):", model_size / 1e6)
print("Time taken to predict:", end_time - start_time)
print("Predicted class:", model.config.id2label[predicted_class_idx])
