import cv2
import numpy as np
import shutil
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model,load_model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from google.colab.patches import cv2_imshow
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform
import tensorflow as tf
from matplotlib import pyplot
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers, activations
import os
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import itertools
#import shutil
import sklearn.metrics as metrics
import seaborn as sns

def load_data():

    train_path="/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Train/"
    val_path="/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Val/"
    test_path="/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Test/"
    class_names=os.listdir(train_path)
    class_names_val=os.listdir(val_path)
    class_names_test=os.listdir(test_path)

    print(class_names)
    print(class_names_test)

    train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Train/",target_size=(224, 224),batch_size=32,shuffle=True,class_mode='binary')
    val_generator = val_datagen.flow_from_directory("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Val/",target_size=(224,224),batch_size=32,shuffle=False,class_mode='binary')
    test_generator = test_datagen.flow_from_directory("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Test/",target_size=(224,224),batch_size=32,shuffle=False,class_mode='binary')

    print(test_generator.class_indices)

    train_benign_list = os.listdir("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Train/Benign") # dir is your directory path
    train_benign_number_files = len(train_benign_list)
    train_benign_number_files

    train_malignant_list = os.listdir("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Train/Malignant") # dir is your directory path
    train_malignant_number_files = len(train_malignant_list)
    train_malignant_number_files

    val_benign_list = os.listdir("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Val/Benign") # dir is your directory path
    val_benign_number_files = len(val_benign_list)
    val_benign_number_files

    val_malignant_list = os.listdir("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Val/Malignant") # dir is your directory path
    val_malignant_number_files = len(val_malignant_list)
    val_malignant_number_files

    test_benign_list = os.listdir("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Test/Benign") # dir is your directory path
    test_benign_number_files = len(test_benign_list)
    test_benign_number_files

    test_malignant_list = os.listdir("/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Test/Malignant") # dir is your directory path
    test_malignant_number_files = len(test_malignant_list)
    test_malignant_number_files

    #Applying Augmentaion on data to avoid overfitting
    train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
    val_datagen = ImageDataGenerator(rescale = 1./255)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set =train_datagen.flow_from_directory('/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Train/',target_size = (224, 224),batch_size = 8,class_mode = 'binary')
    val_set = val_datagen.flow_from_directory('/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Val/',target_size = (224, 224),batch_size = 8,class_mode = 'binary')
    test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/Assignments/Assignment 3/Model_Training_Dataset/Test/',target_size = (224, 224),batch_size = 8,class_mode = 'binary')
    #type(training_set)

    return training_set, val_set, test_set

def conv2d(x,numfilt,filtsz,strides=1,pad='same',act=True,name=None):
  x = Conv2D(numfilt,filtsz,strides,padding=pad,data_format='channels_last',use_bias=False,name=name+'conv2d')(x)
  x = BatchNormalization(axis=3,scale=False,name=name+'conv2d'+'bn')(x)
  if act:
    x = Activation('relu',name=name+'conv2d'+'act')(x)
  return x

def incresA(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,32,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,32,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,32,3,1,pad,True,name=name+'b1_2')
    branch2 = conv2d(x,32,1,1,pad,True,name=name+'b2_1')
    branch2 = conv2d(branch2,48,3,1,pad,True,name=name+'b2_2')
    branch2 = conv2d(branch2,64,3,1,pad,True,name=name+'b2_3')
    branches = [branch0,branch1,branch2]
    mixed = Concatenate(axis=3, name=name + '_concat')(branches)
    filt_exp_1x1 = conv2d(mixed,384,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresB(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,128,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,160,[1,7],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,192,[7,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,1152,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresC(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,192,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,224,[1,3],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,256,[3,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,2048,1,1,pad,False,name=name+'fin1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_saling')([x, filt_exp_1x1])
    return final_lay

def stem_block():
    img_input = Input(shape=(256,256,3))

    x = conv2d(img_input,32,3,2,'valid',True,name='conv1')
    x = conv2d(x,32,3,1,'valid',True,name='conv2')
    x = conv2d(x,64,3,1,'valid',True,name='conv3')

    x_11 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_11'+'_maxpool_1')(x)
    x_12 = conv2d(x,64,3,1,'valid',True,name='stem_br_12')

    x = Concatenate(axis=3, name = 'stem_concat_1')([x_11,x_12])

    x_21 = conv2d(x,64,1,1,'same',True,name='stem_br_211')
    x_21 = conv2d(x_21,64,[1,7],1,'same',True,name='stem_br_212')
    x_21 = conv2d(x_21,64,[7,1],1,'same',True,name='stem_br_213')
    x_21 = conv2d(x_21,96,3,1,'valid',True,name='stem_br_214')

    x_22 = conv2d(x,64,1,1,'same',True,name='stem_br_221')
    x_22 = conv2d(x_22,96,3,1,'valid',True,name='stem_br_222')

    x = Concatenate(axis=3, name = 'stem_concat_2')([x_21,x_22])

    x_31 = conv2d(x,192,3,1,'valid',True,name='stem_br_31')
    x_32 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_32'+'_maxpool_2')(x)
    x = Concatenate(axis=3, name = 'stem_concat_3')([x_31,x_32])

def InceptionResNet():
    num_classes = 2

    #Inception-ResNet-A modules
    x = incresA(x,0.15,name='incresA_1')
    x = incresA(x,0.15,name='incresA_2')
    x = incresA(x,0.15,name='incresA_3')
    x = incresA(x,0.15,name='incresA_4')

    #35 × 35 to 17 × 17 reduction module.
    x_red_11 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_1')(x)

    x_red_12 = conv2d(x,384,3,2,'valid',True,name='x_red1_c1')

    x_red_13 = conv2d(x,256,1,1,'same',True,name='x_red1_c2_1')
    x_red_13 = conv2d(x_red_13,256,3,1,'same',True,name='x_red1_c2_2')
    x_red_13 = conv2d(x_red_13,384,3,2,'valid',True,name='x_red1_c2_3')

    x = Concatenate(axis=3, name='red_concat_1')([x_red_11,x_red_12,x_red_13])

    #Inception-ResNet-B modules
    x = incresB(x,0.1,name='incresB_1')
    x = incresB(x,0.1,name='incresB_2')
    x = incresB(x,0.1,name='incresB_3')
    x = incresB(x,0.1,name='incresB_4')
    x = incresB(x,0.1,name='incresB_5')
    x = incresB(x,0.1,name='incresB_6')
    x = incresB(x,0.1,name='incresB_7')

    #17 × 17 to 8 × 8 reduction module.
    x_red_21 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_2')(x)

    x_red_22 = conv2d(x,256,1,1,'same',True,name='x_red2_c11')
    x_red_22 = conv2d(x_red_22,384,3,2,'valid',True,name='x_red2_c12')

    x_red_23 = conv2d(x,256,1,1,'same',True,name='x_red2_c21')
    x_red_23 = conv2d(x_red_23,256,3,2,'valid',True,name='x_red2_c22')

    x_red_24 = conv2d(x,256,1,1,'same',True,name='x_red2_c31')
    x_red_24 = conv2d(x_red_24,256,3,1,'same',True,name='x_red2_c32')
    x_red_24 = conv2d(x_red_24,256,3,2,'valid',True,name='x_red2_c33')

    x = Concatenate(axis=3, name='red_concat_2')([x_red_21,x_red_22,x_red_23,x_red_24])

    #Inception-ResNet-C modules
    x = incresC(x,0.2,name='incresC_1')
    x = incresC(x,0.2,name='incresC_2')
    x = incresC(x,0.2,name='incresC_3')

    #TOP
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dropout(0.6)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = x
    
    return model

def load_model_InceptionResNet(img_input, x):

    model = Model(img_input,x,name='inception_resnet_v2')
    model.summary()

    dot_img_file = '/content/drive/MyDrive/Assignments/Assignment 4/Inception_ResNet_v2.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    tf.keras.models.save_model(model, '/content/drive/MyDrive/Assignments/Assignment 4/models/Inception-ResNet_v2.h5')

def train_model_InceptionResNet(model, training_set, val_set, test_set):

    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    history = model.fit(training_set,steps_per_epoch = 10,epochs = 400,validation_data = val_set,validation_steps = 5)

def plot_train_test_graph(history, model_version):

    plt.rcParams["figure.figsize"] = (18,9)

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.savefig('/content/drive/MyDrive/Assignments/Assignment 1/TrainTest_LossAccuracy_Graph_' + model_version + '.png')
    pyplot.show()

def classification(model, test_set):

    test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)

    predictions = model.predict_generator(test_set, steps=test_steps_per_epoch)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_set.classes
    class_labels = list(test_set.class_indices.keys())

    return true_classes, predicted_classes, class_labels

def classification_metrics(true_classes, predicted_classes, class_labels, model_version):

    report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

    confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
    print(confusion_matrix)

    plt.rcParams["figure.figsize"] = (6,5)
    ax= plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, fmt='g', ax=ax, annot_kws={"fontsize":18})  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels', fontsize=18)
    ax.set_ylabel('True labels', fontsize=18)
    ax.set_title('Confusion Matrix',fontsize=22)
    ax.xaxis.set_ticklabels(['Benign', 'Malignant'],fontsize=14)
    ax.yaxis.set_ticklabels(['Benign', 'Malignant'],fontsize=14)

    plt.savefig('/content/drive/MyDrive/Paper_Publications_Files/Paper_3/Confusion_Matrix' + model_version + '.png')

