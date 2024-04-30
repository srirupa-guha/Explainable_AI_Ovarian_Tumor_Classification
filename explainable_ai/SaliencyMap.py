#dependencies
import shutil
import os
import random
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

#from vis.utils import utils
from tf_keras_vis.utils.scores import CategoricalScore
from keras.models import load_model

def SaliencyMap(model2, score, x):

    #Create Saliency object
    saliency = Saliency(model2, clone=False)

    subplot_args = {
    'nrows': 1,
    'ncols': 1,
    'figsize': (5, 4),
    'subplot_kw': {'xticks': [], 'yticks': []}
    }

    # Generate saliency map
    saliency_map = saliency(score, x, smooth_samples=20, smooth_noise=0.2)
    saliency_map = normalize(saliency_map)

    f, ax = plt.subplots(**subplot_args)
    ax.imshow(saliency_map[0], cmap='Reds')
    plt.tight_layout()
    plt.show()

def run_SaliencyMap(img_path, model):

    #img_path = '/content/drive/MyDrive/Academic_Courses_and_ML_Projects/Paper_Publications_Files/Paper_3/Annotated_Images/Annotated_Dataset/Malignant/Folder_21_3_1_img11.jpg'
    
    img = tf.keras.preprocessing.image.load_img(img_path,target_size=(224,224))
    x = img_to_array(img)  # Numpy array with shape (300, 300, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 300, 300, 3)

    plt.imshow(img)

    layer_name = 'fc_3'
    # model = load_model('/content/drive/MyDrive/Academic_Courses_and_ML_Projects/Paper_Publications_Files/Paper_3/Ovarian_Image_classification_ResNet60.h5')
    # model.summary()

    model2 = model
    model2.layers[-1].activation = tf.keras.activations.linear
    score = CategoricalScore([0])

    SaliencyMap(model2, score, x)
