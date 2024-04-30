import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import math
import pylab
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from numpy.random import permutation
from keras.optimizers import SGD
import cv2
from keras.models import load_model

def Occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride, model):

    image = cv2.imread(image_path)
    im = cv2.resize(image, (224, 224)).astype(np.float32)

    #im = (im - 0.5)*2
    #im = np.expand_dims(im, axis=0)

    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    #im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    out = model.predict(im)
    out = out[0]
    # Getting the index of the winning class:
    m = max(out)
    index_object = [i for i, j in enumerate(out) if j == m]
    height, width, _ = image.shape
    output_height = int(math.ceil((height-occluding_size) / occluding_stride + 1))
    output_width = int(math.ceil((width-occluding_size) / occluding_stride + 1))
    heatmap = np.zeros((output_height, output_width))

    for h in range(output_height):
        for w in range(output_width):
            # Occluder region:
            h_start = h * occluding_stride
            w_start = w * occluding_stride
            h_end = min(height, h_start + occluding_size)
            w_end = min(width, w_start + occluding_size)
            # Getting the image copy, applying the occluding window and classifying it again:
            input_image = copy.copy(image)
            input_image[h_start:h_end, w_start:w_end,:] =  occluding_pixel
            im = cv2.resize(input_image, (224, 224)).astype(np.float32)
            #im = im.transpose((2,0,1))
            im = np.expand_dims(im, axis=0)
            out = model.predict(im)
            out = out[0]
            print('scanning position (%s, %s)'%(h,w))
            # It's possible to evaluate the VGG-16 sensitivity to a specific object.
            # To do so, you have to change the variable "index_object" by the index of
            # the class of interest. The VGG-16 output indices can be found here:
            # https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
            prob = (out[index_object])
            heatmap[h,w] = prob

    f = pylab.figure()
    f.add_subplot(1, 2, 1)  # this line outputs images side-by-side
    ax = sns.heatmap(heatmap,xticklabels=False, yticklabels=False)
    f.add_subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()
    print ( 'Object index is %s'%index_object)

def run_Occlusion(image_path, model):
    
    #image_path = '/content/drive/MyDrive/Academic_Courses_and_ML_Projects/Paper_Publications_Files/Paper_3/Annotated_Images/Annotated_Dataset/Malignant/Folder_21_3_1_img11.jpg'
    im = cv2.imread(image_path)
    print(im.shape)

    occluding_size = 5
    occluding_pixel = 0
    occluding_stride = 1

    #model = load_model('/content/drive/MyDrive/Academic_Courses_and_ML_Projects/Paper_Publications_Files/Paper_3/Ovarian_Image_classification_ResNet60.h5')

    Occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride, model)