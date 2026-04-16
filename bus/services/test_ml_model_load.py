# tensorflow가 필요할 경우에만 import
# import tensorflow as tf
import keras
import os
from django.conf import settings

MODEL1_PATH = os.path.join(settings.BASE_DIR, "ml_models", "test_model1.keras")
MODEL2_PATH = os.path.join(settings.BASE_DIR, "ml_models", "test_model2.keras")
MODEL3_PATH = os.path.join(settings.BASE_DIR, "ml_models", "test_model3.keras")
print("loading ML model1...")
ml_model_1 = keras.models.load_model(MODEL1_PATH)
print("loading done.")
print("loading ML model2...")
ml_model_2 = keras.models.load_model(MODEL2_PATH)
print("loading done.")
print("loading ML model3...")
ml_model_3 = keras.models.load_model(MODEL3_PATH)
print("loading done.")