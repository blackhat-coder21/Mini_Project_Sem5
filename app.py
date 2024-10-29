import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import os
import imageio
import cv2
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import ipywidgets as widgets
from deepface import DeepFace
from ultralytics import YOLO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import requests
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Normalization
from bs4 import BeautifulSoup
import string
import traceback
import time
from streamlit_lottie import st_lottie
import json
import pickle

# Function to load Lottie animation from a file
def load_lottie_animation(animation_path):
    with open(animation_path, "r") as f:
        return json.load(f)

# Load the animations
red_skull_animation = load_lottie_animation("wrong.json")
happy_emoji_animation = load_lottie_animation("tick.json")


# To store the cropped face
cropped_face = None

def process_uploaded_image():
    global uploaded_image, uploaded_image_path, cropped_face

    if uploaded_image is None or uploaded_image_path is None:
        print("No image uploaded.")
        return

    # Ensure the uploaded image is a PIL Image
    if not isinstance(uploaded_image, Image.Image):
        uploaded_image = Image.open(uploaded_image_path)

    # Convert the PIL Image to a NumPy array with proper type
    img_array = np.array(uploaded_image).astype(np.uint8)

    # Convert from RGB (PIL format) to BGR (OpenCV format)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Face detection using Haar Cascade
    face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_engine.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        max_area = 0
        max_cordinate = None

        # Find the largest face
        for (x, y, w, h) in faces:
            area = h * w
            if area > max_area:
                max_area = area
                max_cordinate = (x, y, w, h)

        # Crop and resize the largest face
        if max_cordinate is not None:
            cropped = img[max_cordinate[1]:max_cordinate[1] + max_cordinate[3], max_cordinate[0]:max_cordinate[0] + max_cordinate[2]]
            cropped_face = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
            print("Cropped face stored in global variable.")
        else:
            print("No face detected in the image.")
    else:
        print("No faces found.")




# To store the analysis results
result = None

def analyze_single_image():
    global cropped_face, result

    if cropped_face is None:
        print("No cropped face image available.")
        return

    try:
        # Analyze the image for emotion, race, gender, and age
        predictions = DeepFace.analyze(cropped_face, actions=["emotion", "race", "gender", "age"], enforce_detection=False)

        # Check if predictions is a list and has at least one item
        if isinstance(predictions, list) and len(predictions) > 0:
            dominant_emotion = predictions[0]['dominant_emotion']
            dominant_race = predictions[0]['dominant_race']
            gender_scores = predictions[0]['gender']

            # Extract the dominant gender based on the scores
            if gender_scores['Woman'] > gender_scores['Man']:
                dominant_gender = 'Female'
            else:
                dominant_gender = 'Male'

            age = predictions[0]['age']

            # Bucket the age
            if 13 <= age <= 17:
                agebucket = '13-17 years'
            elif 18 <= age <= 24:
                agebucket = '18-24 years'
            elif 25 <= age <= 34:
                agebucket = '25-34 years'
            elif 35 <= age <= 44:
                agebucket = '35-44 years'
            elif 45 <= age <= 54:
                agebucket = '45-54 years'
            elif 55 <= age <= 64:
                agebucket = '55-64 years'
            elif age > 64:
                agebucket = 'above 65 years'
            else:
                agebucket = 'NA'

            # Store the analysis results in the global variable
            result = {
                'Dominant Emotion': dominant_emotion,
                'Dominant Race': dominant_race,
                'Dominant Gender': dominant_gender,
                'Age': age,
                'Age Group': agebucket
            }

            print("Analysis Results:")
            print(result)

        else:
            result = 'NA'

    except Exception as e:
        result = 'NA'
        traceback.print_exc()


# To store the results from the detection function
detected_objects = None

def detect_objects_with_unique_labels(uploaded_image_path):
    global detected_objects
    detection_results = None
    # Check if the uploaded image path is set
    if uploaded_image_path is None:
        print("No image path provided.")
        return
    
    # Load the pre-trained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Perform object detection
    results = model(uploaded_image_path)

    # Dictionary to keep count of each detected object
    object_count = {}
    detected_objects = []

    # Extract detected object names with unique labels
    for result in results:
        for obj in result.boxes.data:
            # Get the class id and name of the detected object
            class_id = int(obj[5].item())
            class_name = model.names[class_id]

            # Update the count for this object
            if class_name in object_count:
                object_count[class_name] += 1
            else:
                object_count[class_name] = 1

            # Create a unique label by appending the count
            unique_label = f"{class_name}_{object_count[class_name]}"
            detected_objects.append(unique_label)

    # Store the detected objects and results in the global variable
    detection_results = (detected_objects, results)

    # Print the detected objects for debugging
    print("Detected Objects with Unique Labels:", detected_objects)



model = None
processor = None
is_qwen_model_loaded = False

def load_Qwen_model():
    global model, processor, is_qwen_model_loaded
    if not is_qwen_model_loaded:
        # Load the Qwen model and processor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype="auto",
            device_map="auto",
        )

        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct"
        )
        is_qwen_model_loaded = True
        print("Qwen model loaded.")

# To store the extracted text
extracted_text = None

def extract_text_from_image(uploaded_image_path):
    global extracted_text
    
    # Open the image using Pillow
    image = Image.open(uploaded_image_path)

    # Prepare the messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": "Please extract the text from the image."
                },
            ],
        }
    ]

    # Apply chat template and prepare inputs
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Generate output
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    # Decode output text
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Replace \n with space
    output_text = [text.replace('\n', ' ') for text in output_text]

    # Combine all extracted texts into a single string
    final_output = ' '.join(output_text)
    
    # Store the result in the global variable
    extracted_text = final_output

    # Print the final output for debugging
    print("Extracted Text:", extracted_text)
    

# Loading sentiments analysis model and checking if already loaded or not
model_sentiments = None
is_sentiments_model_loaded = False

def load_sentiments_model():
    global model_sentiments, is_sentiments_model_loaded
    if not is_sentiments_model_loaded:
        # Recreate the model architecture
        model_sentiments = Sequential()
        model_sentiments.add(LSTM(256, input_shape=(61, 1), return_sequences=True))
        model_sentiments.add(LSTM(128))
        model_sentiments.add(Normalization())
        model_sentiments.add(Dense(64, activation='relu'))
        model_sentiments.add(Dense(3, activation='softmax'))

        # Load the model weights
        model_sentiments.load_weights('sentiment_analysis_model.h5')
        is_sentiments_model_loaded = True
        print("Sentiments model loaded.")


# Loading the pickle file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# To store sentiments analysis result 
predicted_sentiments = []


# To preprocess the sentiments analysis
def preprocess_tweet(tweet):
    # Remove HTML tags
    soup = BeautifulSoup(tweet, 'html.parser')
    tweet = soup.get_text()
    
    # Remove punctuation
    punctuation = string.punctuation
    tweet = tweet.translate(str.maketrans('', '', punctuation))
    
    # Lowercase the text
    tweet = tweet.lower()
    
    # Tokenize the tweet
    sequence = tokenizer.texts_to_sequences([tweet])
    
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=model_sentiments.input_shape[1])
    
    return padded_sequence

# To perform sentiment analysis on extracted text
def analyze_sentiment():
    global extracted_text, predicted_sentiments

    # Check if extracted_text is None
    if extracted_text is None:
        print("No text extracted for sentiment analysis.")
        return

    # Preprocess the extracted text
    preprocessed_tweet = preprocess_tweet(extracted_text)

    # Make predictions
    prediction = model_sentiments.predict(preprocessed_tweet)

    # Convert prediction to label
    predicted_label = np.argmax(prediction)

    # Map the numerical label back to original sentiment labels
    label_map = {0: 'bad', 1: 'good', 2: 'neutral'}
    sentiment = label_map[predicted_label]

    # Store the predicted sentiment in the global variable
    predicted_sentiments.append(sentiment)

    # Display the results
    print(f'Extracted Text: "{extracted_text}" -> Sentiment: {sentiment}')

    
# To display prediction results with animation
def display_prediction_with_animation(prediction_result, reason):
    if prediction_result is not None:
        result_text = "Non-Hateful" if prediction_result == 0 else "Hateful"
        
        # Display appropriate animation based on the prediction
        if result_text == "Hateful":
            st_lottie(red_skull_animation, height=200, key="red_skull")
        else:
            st_lottie(happy_emoji_animation, height=200, key="happy_emoji")
        
        # To show the prediction result and reason
        st.write(f"Prediction: {result_text}")
        st.write(f"Reason: {reason}")
    else:
        st.write("Prediction could not be made.")


# To store the prediction result
prediction_result = None
reason = None

def hateful_meme_detection(image_path):
    global prompt, uploaded_image_path, prediction_result, reason, predicted_sentiments
    
    # load_Qwen_model()
    # load_sentiments_model()
    # process_uploaded_image()
    # analyze_single_image()
    # detect_objects_with_unique_labels(image_path)
    # extract_text_from_image(image_path)
    # analyze_sentiment()

    # True implies "image_path" needed
    functions = [
        ("Load Qwen Model", load_Qwen_model, False),
        ("Load Sentiments Model", load_sentiments_model, False),
        ("Process Uploaded Image", process_uploaded_image, False),
        ("Analyze Single Image", analyze_single_image, False),
        ("Detect Objects", detect_objects_with_unique_labels, True),
        ("Extract Text", extract_text_from_image, True),
        ("Analyze Sentiment", analyze_sentiment, False),
    ]

    # To display the steps in a graphical layout
    st.markdown("### Detection Workflow")

    # Step 1: Model Loading (Qwen and Sentiment Models)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Load Qwen Model")
        start_time = time.time()
        with st.spinner("Loading Qwen Model..."):
            load_Qwen_model()
        elapsed_time = time.time() - start_time
        st.success(f"Qwen Model Loaded in {elapsed_time:.2f} seconds")

    with col2:
        st.markdown("#### Load Sentiments Model")
        start_time = time.time()
        with st.spinner("Loading Sentiments Model..."):
            load_sentiments_model()
        elapsed_time = time.time() - start_time
        st.success(f"Sentiments Model Loaded in {elapsed_time:.2f} seconds")

    # Step 2: To analyze Image
    st.markdown("#### Analyze Image")
    start_time = time.time()
    with st.spinner("Analyzing Image..."):
        analyze_single_image()
    elapsed_time = time.time() - start_time
    st.success(f"Image Analyzed in {elapsed_time:.2f} seconds")

    # Step 3: To detect Objects and Extract Text
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Detect Objects")
        start_time = time.time()
        with st.spinner("Detecting Objects..."):
            detect_objects_with_unique_labels(image_path)
        elapsed_time = time.time() - start_time
        st.success(f"Objects Detected in {elapsed_time:.2f} seconds")

    with col2:
        st.markdown("#### Extract Text")
        start_time = time.time()
        with st.spinner("Extracting Text..."):
            extract_text_from_image(image_path)
        elapsed_time = time.time() - start_time
        st.success(f"Text Extracted in {elapsed_time:.2f} seconds")

    # Step 4: To analyze Sentiment
    st.markdown("#### Analyze Sentiment")
    start_time = time.time()
    with st.spinner("Analyzing Sentiment..."):
        analyze_sentiment()
    elapsed_time = time.time() - start_time
    st.success(f"Sentiment Analysis Completed in {elapsed_time:.2f} seconds")


    # Creating final prompt to pass to model
    context_parts = []
    
    if detected_objects:
        context_parts.append(f"Detected objects: {detected_objects}")

    if result is not None:
        if result.get('Dominant Emotion'):
            context_parts.append(f"Dominant emotion: {result['Dominant Emotion']}")
        if result.get('Dominant Race'):
            context_parts.append(f"Race: {result['Dominant Race']}")
        if result.get('Dominant Gender'):
            context_parts.append(f"Gender: {result['Dominant Gender']}")
        if result.get('Age Group'):
            context_parts.append(f"Age group: {result['Age Group']}")

    if extracted_text:
        context_parts.append(f"Text in meme: '{extracted_text}'")
    if predicted_sentiments:
        context_parts.append(f"Sentiment: {predicted_sentiments}")

    prompt = ""
    prompt = ". ".join(context_parts) + "." if context_parts else "No relevant information available."
    
    # Step 1: Load the image
    global_uploaded_image = Image.open(image_path)

    # Step 2: Prepare the global prompt
    data = f"Please predict whether the content in the meme is hateful or non-hateful. Context: {prompt}"

    st.markdown("#### Loading Final Model")
    start_time = time.time()
    with st.spinner("Analyzing Meme..."):
        # Step 3: Prepare the messages for the model, including the global prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": data,
                    },
                ],
            }
        ]

        # Step 4: Apply the chat template and create inputs for the model
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[global_uploaded_image],
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Step 5: Generate the output from the model
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Step 6: Replace '\n' with a space in the output text
        output_text = [text.replace('\n', ' ') for text in output_text]

        # Step 7: Store the final output in global variables
        reason = ' '.join(output_text)

        # To determine prediction based on the final output
        if "non-hateful" in reason.lower():
            prediction_result = 0  # Non-hateful
        else:
            prediction_result = 1  # Hateful

        # Step 8: Print the results
        print(f"Prediction: {'Non-Hateful' if prediction_result == 0 else 'Hateful'},\nReason: {reason}\n")
    

    # Final output
    elapsed_time = time.time() - start_time
    st.success(f"Hateful Meme Detection completed successfully!!! in {elapsed_time:.2f} seconds")
    st.balloons()



# Streamlit UI Part
st.title("Hateful Meme Detection")

uploaded_image = st.file_uploader("Upload a Meme Image", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    # 3x1 grid layout
    col1, col2= st.columns(2)

    # To display the uploaded image in the first column
    with col1:
        st.image(uploaded_image, caption='Uploaded Meme', width=250)
    
    # Save uploaded image to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_image.read())
    uploaded_image_path = "temp_image.jpg"

    process_uploaded_image()
    detect_objects_with_unique_labels(uploaded_image_path)
    analyze_single_image()

    # To display the extracted face in the second column
    with col2:
        if cropped_face is not None:
            st.image(cropped_face, caption='Extracted Face', width=250)
        else:
            st.write("No face detected.")

    # Run the prediction
    hateful_meme_detection(uploaded_image_path)
    
    # Display the results
    st.header("Prediction Result")
    display_prediction_with_animation(prediction_result, reason)
