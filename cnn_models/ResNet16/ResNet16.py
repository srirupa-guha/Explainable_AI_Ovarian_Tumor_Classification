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
import matplotlib.pyplot as plt
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

def identity_block(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet16(input_shape=(224, 224, 3)):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet16')

    return model

def load_model_ResNet16():

    base_model = ResNet16(input_shape=(224, 224, 3))
    headModel = base_model.output
    headModel = Flatten()(headModel)
    headModel=Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel)
    headModel=Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
    headModel = Dense( 1,activation='sigmoid', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)

    model = Model(inputs=base_model.input, outputs=headModel)
    model.summary()

    dot_img_file = '/content/drive/MyDrive/Assignments/Assignment 3/ResNet16.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    return model

def train_model_ResNet16(model, training_set, val_set, test_set):

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(training_set,steps_per_epoch = 10,epochs = 200,validation_data = val_set,validation_steps = 5)

    return history

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



