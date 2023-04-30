from django.shortcuts import render

# Create your views here.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from rest_framework.views import APIView
from rest_framework.response import Response
import time
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

class MLModel(APIView):
    def post(self, request, format = None):
        html_code = request.data["html_code"]
        css_code = request.data["css_code"]
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_window_size(1419, 649)
        driver.get("about:blank")
        driver.execute_script("document.write('<html><head><style>" + css_code + "</style></head><body>" + html_code + "</body></html>');")
        time.sleep(3)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # driver.save_screenshot("screenshotw4w.png")

        
        screenshot = driver.get_screenshot_as_png()
        a = BytesIO(screenshot)
        # rendered_image = Image.open(BytesIO(screenshot))

        # Load pre-trained ResNet50 model
        model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Function to preprocess image for ResNet50 model
        def preprocess(img_path):
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return x

        # Function to compute similarity score between two images
        def image_similarity(image1_path, image2_path):
            # Preprocess images
            img1 = preprocess(image1_path)
            img2 = preprocess(image2_path)

            # Get features from pre-trained model
            features1 = model.predict(img1)
            features2 = model.predict(img2)

            # Flatten features and compute cosine similarity
            features1 = features1.flatten()
            features2 = features2.flatten()
            similarity_score = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
            return similarity_score

        try:
            image1_path = a
            image2_path = "test.png"
            similarity_score = image_similarity(image1_path, image2_path)
            result = round(similarity_score*100)
            
            return Response({'message' : str(result)})
            
        except Exception as e:
            return Response({'error' : str(e)})