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
import numpy as np
import pandas as pd
import requests
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor
import time
import os
import sys 
from tqdm import tqdm

def compute_logits(image_file_path, save_directory, batch_size=16):

    print("Loading model.....")
    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
    print("Model loaded successfully!")
    print("Loading feature extractor...")
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
    print("Feature extractor loaded successfully!")
    model_size = model.num_parameters()

    #output class num 
    num_classes = model.config.num_labels
    print("Number of classes:", num_classes)
    
    logits_list = []
    image_class_list = []

    image_files = [os.path.join(image_file_path, file) for file in os.listdir(image_file_path) if file.endswith('.JPEG') or file.endswith('.jpg')]

    overall_start_time = time.time()


  
    with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            images = [Image.open(file).convert("RGB") for file in batch_files]
            inputs = feature_extractor(images=images, return_tensors="pt")



            outputs = model(**inputs)
            logits = outputs.logits
            
           

            predicted_class_indices = logits.argmax(-1).tolist()
            image_classes = [model.config.id2label[idx] for idx in predicted_class_indices]

            logits_list.extend(logits.tolist())
            image_class_list.extend(image_classes)
            

           
            pbar.update(len(batch_files))
        
    overall_end_time = time.time()
    overall_total_time = overall_end_time - overall_start_time

    average_inference_time = overall_total_time / len(image_files)
    print("Average inference time per image:", average_inference_time, "seconds")

    # Save the logits, image classes, and inference times to a CSV file
    df = pd.DataFrame({
        'Image Class': image_class_list,
        'Logits': logits_list,
    })
    df.to_csv(save_directory + '/logits_output.csv', index=False)

    #save the model size and the total inference time to a txt file
    model_size = model.num_parameters()
    model_size_mb = model_size / 1e6

    with open(save_directory + '/model_size_and_inference_time.txt', 'w') as f:
        f.write(f'Model size (in parameters): {model_size}\n')
        f.write(f'Model size (in MB): {model_size_mb}\n')
        f.write(f'Total inference time for all images: {overall_total_time} seconds\n')
        f.write(f'Average inference time per image: {average_inference_time} seconds\n')
    
    # Print the model size and total inference time
    print("Model size (in parameters):", model_size)
    print("Model size (in MB):", model_size_mb)
    print("Total inference time for all images:", overall_total_time, "seconds")
    return image_class_list 

def save_all_extracted_features(image_file_path, save_directory, batch_size=16):
    print("Loading feature extractor...")
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
    print("Feature extractor loaded successfully!")
    
    # List image files (case-insensitive for file extensions)
    image_files = [os.path.join(image_file_path, file) 
                   for file in os.listdir(image_file_path)
                   if file.lower().endswith(('.jpeg', '.jpg'))]

    all_features = []
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        # Open and convert images
        images = [Image.open(file).convert("RGB") for file in batch_files]
        # Process images to obtain pixel values
        inputs = feature_extractor(images=images, return_tensors="pt")
        features = inputs['pixel_values'].numpy()  # Shape: [batch_size, channels, height, width]
        all_features.append(features)
        print(f"Processed batch {i // batch_size + 1}/{total_batches}")

    # Concatenate all batches along the batch dimension
    all_features = np.concatenate(all_features, axis=0)
    save_path = os.path.join(save_directory, 'all_features.npy')
    np.save(save_path, all_features)
    print(f"Extracted features shape: {all_features.shape}")
    print(f"Saved all extracted features to {save_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python Getting_Main_Model_Logits.py <image_file_path> <save_directory> <batch_size> (optional)")
        print("Example: python Getting_Main_Model_Logits.py /path/to/images /path/to/save_directory 16")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print(f"Error: The directory {sys.argv[1]} does not exist.")
        sys.exit(1)
    if not os.path.exists(sys.argv[2]):
        print(f"Error: The directory {sys.argv[2]} does not exist.")
        sys.exit(1)
    if len(sys.argv) == 4:
        try:
            batch_size = int(sys.argv[3])
        except ValueError:
            print("Error: Batch size must be an integer.")
            sys.exit(1)
    else:
        batch_size = 16
    
    image_file_path = sys.argv[1]
    save_directory = sys.argv[2]

    



    

    # Call the function to compute logits
    #compute_logits(image_file_path, save_directory, batch_size)
    save_all_extracted_features(image_file_path, save_directory, batch_size)

if __name__ == "__main__":
    main()
    
