import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from deepface import DeepFace
from ultralytics import YOLO

# Global variables
model = None
processor = None
is_qwen_model_loaded = False

def load_Qwen_model():
    global model, processor, is_qwen_model_loaded
    if not is_qwen_model_loaded:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype="auto",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        is_qwen_model_loaded = True
        print("Qwen model loaded.")

def build_prompt(img_id, data_csv):
    img_id = str(img_id)
    img_name = img_id.lstrip('0')  # Remove leading zeros
    result = data_csv[data_csv['id'] == int(img_name)]

    if result.empty:
        return "No relevant information available."

    row = result.iloc[0]
    context_parts = []

    if pd.notna(row['detected_objects']):
        context_parts.append(f"Detected objects: {row['detected_objects']}")
    if pd.notna(row['Dominant Emotion']):
        context_parts.append(f"Dominant emotion: {row['Dominant Emotion']}")
    if pd.notna(row['Dominant Race']):
        context_parts.append(f"Race: {row['Dominant Race']}")
    if pd.notna(row['gender']):
        context_parts.append(f"Gender: {row['gender']}")
    if pd.notna(row['age_group']):
        context_parts.append(f"Age group: {row['age_group']}")
    if pd.notna(row['text']):
        context_parts.append(f"Text in meme: '{row['text']}'")
    if pd.notna(row['sentiment_analysis']):
        context_parts.append(f"Sentiment: {row['sentiment_analysis']}")

    return ". ".join(context_parts) + "." if context_parts else "No relevant information available."

def hateful_meme_detection(image_path, img_name, data_csv):
    global prompt, prediction_result, reason
    load_Qwen_model()
    
    prompt = build_prompt(img_name, data_csv)
    print(prompt)
    global_uploaded_image = Image.open(image_path)

    data = f"Please predict whether the content in the meme is hateful or non-hateful. Context: {prompt}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": data},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[global_uploaded_image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    reason = ' '.join(output_text).replace('\n', ' ')

    prediction_result = 1 if "non-hateful" not in reason.lower() else 0
    print(f"Prediction: {'Non-Hateful' if prediction_result == 0 else 'Hateful'}, Reason: {reason}")

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_predictions_to_csv(predictions, output_file):
    df = pd.DataFrame(predictions, columns=['id', 'actual_label', 'predicted_label', 'result', 'reason'])
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def analyze_results(predictions):
    actual_labels = [entry['actual_label'] for entry in predictions]
    predicted_labels = [entry['predicted_label'] for entry in predictions]

    report = classification_report(actual_labels, predicted_labels)
    print("Classification Report:\n", report)
    accuracy = accuracy_score(actual_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.2f}")
    precision = precision_score(actual_labels, predicted_labels, pos_label=1)
    print(f"Precision: {precision:.2f}")
    recall = recall_score(actual_labels, predicted_labels, pos_label=1)
    print(f"Recall: {recall:.2f}")

    cm = confusion_matrix(actual_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Hateful', 'Hateful'], yticklabels=['Non-Hateful', 'Hateful'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def process_memes_and_predict(file_path, img_folder, output_csv, data_path):
    data = load_data(file_path)
    data_csv = pd.read_csv(data_path)
    predictions = []

    start_time = time.time()
    for entry in tqdm(data, desc="Processing Memes", unit="image"):
        img_id = entry['id']
        actual_label = entry['label']
        img_name = str(img_id).zfill(5) + ".png"
        img_path = os.path.join(img_folder, img_name)
        img_csv = str(img_id).zfill(5)

        hateful_meme_detection(img_path, img_csv, data_csv)
        predicted_label = prediction_result
        prediction_entry = {
            'id': img_id,
            'actual_label': actual_label,
            'predicted_label': predicted_label,
            'result': "hateful" if predicted_label == 1 else "non-hateful",
            'reason': reason
        }
        predictions.append(prediction_entry)

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds.")
    save_predictions_to_csv(predictions, output_csv)
    analyze_results(predictions)

if __name__ == "__main__":
    train_file = 'train.jsonl'
    img_folder = 'images/img'
    output_csv_file = 'results.csv'
    data_path = 'final_datasets.csv'

    process_memes_and_predict(train_file, img_folder, output_csv_file, data_path)
