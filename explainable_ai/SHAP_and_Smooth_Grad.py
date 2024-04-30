# Import all the necessary libraries

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import cv2

# Installing Shap libary; this needs to be reinstalled for every run
import shap

from keras.models import load_model

print(tf.__version__) # 2.15.0
print(shap.__version__) # 0.44.0

def SHAP(x_test, model, sample_image_batch):

    background = shap.utils.sample(x_test, 100)

    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer(model, background)

    shap_values = explainer.shap_values(sample_image_batch)
    print("sample_image_batch.shape: \t", sample_image_batch.shape, "\n")
    print("len(shap_values): \t", len(shap_values), "\n")
    sample_image_batch.shape

    # Visualize the SHAP values
    shap.image_plot(shap_values, sample_image_batch)

    return shap_values, explainer

def Smooth_Grad(NUM_ITERATIONS, shap_values_list, sample_image, explainer):

    for _ in range(NUM_ITERATIONS):
        perturbed_image = sample_image + np.random.normal(loc=0, scale=0.1, size=sample_image.shape)

        #print("perturbed_image.shape: \t", perturbed_image.shape, "\n")
        #print("np.min(perturbed_image): \t", np.min(perturbed_image), "\n")
        #print("np.max(perturbed_image): \t", np.max(perturbed_image), "\n")
        perturbed_image = np.clip(perturbed_image, 0, 1)
        #print("-----------------------------------------------------------------------------------")
        #print("np.min(perturbed_image): \t", np.min(perturbed_image), "\n")
        #print("np.max(perturbed_image): \t", np.max(perturbed_image), "\n\n")
        perturbed_image_batch = np.expand_dims(perturbed_image, axis=0)
        print("perturbed_image_batch: \n", perturbed_image_batch.shape, "\n")
        shap_values = explainer.shap_values(perturbed_image_batch)
        shap.image_plot(shap_values, perturbed_image_batch)
        shap_values_list.append(shap_values)

def run_SHAP(image_path, model):

    #images_dir = 'content/drive/MyDrive/Academic_Courses_and_ML_Projects/Paper_Publications_Files/Paper_3/Annotated_Images/Annotated_Dataset/Benign'
    #model = load_model('/content/drive/MyDrive/Academic_Courses_and_ML_Projects/Paper_Publications_Files/Paper_3/Ovarian_Image_classification_ResNet60.h5')
    #x_test = cv2.imread('/content/drive/MyDrive/Academic_Courses_and_ML_Projects/Paper_Publications_Files/Paper_3/Annotated_Images/Annotated_Dataset/Malignant/69.jpg')
    x_test = cv2.imread(image_path)

    x_test = cv2.resize(x_test, (224, 224))
    x_test = np.expand_dims(x_test, axis=0)

    # Feature Normalization (to [0,1])
    x_test = x_test.astype('float32') / 255.0

    # Choose a random sample from the test set to explain (against the background)

    sample_index = np.random.randint(0, x_test.shape[0])
    sample_image = x_test[sample_index]
    print("sample_index: \t:", sample_index, "\n")
    print("sample_image.shape: \t", sample_image.shape, "\n")

    sample_image_batch = np.expand_dims(sample_image, axis=0)  # Add batch dimension

    shap_values, explainer = SHAP(x_test, model, sample_image_batch)

    return sample_image, explainer

def run_Smooth_Grad(sample_image, explainer):

    # Now implementing the SmoothGrad algorithm

    NUM_ITERATIONS = 3 # equal to number of images that we wish to generate
    shap_values_list = []

    Smooth_Grad(NUM_ITERATIONS, shap_values_list, sample_image, explainer)





