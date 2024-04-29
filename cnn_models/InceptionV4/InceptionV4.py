# Implementation of Inception-v4 architecture

# Importing the libraries
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot
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

def conv_block(x, nb_filter, nb_row, nb_col, padding = "same", strides = (1, 1), use_bias = False):
    '''Defining a Convolution block that will be used throughout the network.'''

    x = Conv2D(nb_filter, (nb_row, nb_col), strides = strides, padding = padding, use_bias = use_bias)(x)
    x = BatchNormalization(axis = -1, momentum = 0.9997, scale = False)(x)
    x = Activation("relu")(x)

    return x

def stem(input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = conv_block(input, 32, 3, 3, strides = (2, 2), padding = "same") # 149 * 149 * 32
    x = conv_block(x, 32, 3, 3, padding = "same") # 147 * 147 * 32
    x = conv_block(x, 64, 3, 3) # 147 * 147 * 64

    x1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(x)
    x2 = conv_block(x, 96, 3, 3, strides = (2, 2), padding = "same")

    x = concatenate([x1, x2], axis = -1) # 73 * 73 * 160

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding = "same")

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding = "same")

    x = concatenate([x1, x2], axis = -1) # 71 * 71 * 192

    x1 = conv_block(x, 192, 3, 3, strides = (2, 2), padding = "same")

    x2 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(x)

    x = concatenate([x1, x2], axis = -1) # 35 * 35 * 384

    return x

def inception_A(input):
    '''Architecture of Inception_A block which is a 35 * 35 grid module.'''

    a1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    a1 = conv_block(a1, 96, 1, 1)

    a2 = conv_block(input, 96, 1, 1)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = conv_block(input, 64, 1, 1)
    a4 = conv_block(a4, 96, 3, 3)
    a4 = conv_block(a4, 96, 3, 3)

    merged = concatenate([a1, a2, a3, a4], axis = -1)

    return merged

def inception_B(input):
    '''Architecture of Inception_B block which is a 17 * 17 grid module.'''

    b1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    b1 = conv_block(b1, 128, 1, 1)

    b2 = conv_block(input, 384, 1, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 256, 7, 1)

    b4 = conv_block(input, 192, 1, 1)
    b4 = conv_block(b4, 192, 7, 1)
    b4 = conv_block(b4, 224, 1, 7)
    b4 = conv_block(b4, 224, 7, 1)
    b4 = conv_block(b4, 256, 1, 7)

    merged = concatenate([b1, b2, b3, b4], axis = -1)

    return merged

def inception_C(input):
    '''Architecture of Inception_C block which is a 8 * 8 grid module.'''

    c1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    c1 = conv_block(c1, 256, 1, 1)

    c2 = conv_block(input, 256, 1, 1)

    c3 = conv_block(input, 384, 1, 1)
    c31 = conv_block(c2, 256, 1, 3)
    c32 = conv_block(c2, 256, 3, 1)
    c3 = concatenate([c31, c32], axis = -1)

    c4 = conv_block(input, 384, 1, 1)
    c4 = conv_block(c3, 448, 3, 1)
    c4 = conv_block(c3, 512, 1, 3)
    c41 = conv_block(c3, 256, 1, 3)
    c42 = conv_block(c3, 256, 3, 1)
    c4 = concatenate([c41, c42], axis = -1)

    merged = concatenate([c1, c2, c3, c4], axis = -1)

    return merged

def reduction_A(input, k = 192, l = 224, m = 256, n = 384):
    '''Architecture of a 35 * 35 to 17 * 17 Reduction_A block.'''

    ra1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(input)

    ra2 = conv_block(input, n, 3, 3, strides = (2, 2), padding = "same")

    ra3 = conv_block(input, k, 1, 1)
    ra3 = conv_block(ra3, l, 3, 3)
    ra3 = conv_block(ra3, m, 3, 3, strides = (2, 2), padding = "same")

    merged = concatenate([ra1, ra2, ra3], axis = -1)

    return merged

def reduction_B(input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_B block.'''

    rb1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(input)

    rb2 = conv_block(input, 192, 1, 1)
    rb2 = conv_block(rb2, 192, 3, 3, strides = (2, 2), padding = "same")

    rb3 = conv_block(input, 256, 1, 1)
    rb3 = conv_block(rb3, 256, 1, 7)
    rb3 = conv_block(rb3, 320, 7, 1)
    rb3 = conv_block(rb3, 320, 3, 3, strides = (2, 2), padding = "same")

    merged = concatenate([rb1, rb2, rb3], axis = -1)

    return merged

def inception_v4(nb_classes = 1, load_weights = True):
    '''Creates the Inception_v4 network.'''

    init = Input((299, 299, 3)) # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering

    # Input shape is 299 * 299 * 3
    x = stem(init) # Output: 35 * 35 * 384

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)
        # Output: 35 * 35 * 384

    # Reduction A
    x = reduction_A(x, k = 192, l = 224, m = 256, n = 384) # Output: 17 * 17 * 1024

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)
        # Output: 17 * 17 * 1024

    # Reduction B
    x = reduction_B(x) # Output: 8 * 8 * 1536

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)
        # Output: 8 * 8 * 1536

    # Average Pooling
    x = AveragePooling2D((8, 8))(x) # Output: 1536

    # Dropout
    x = Dropout(0.2)(x) # Keep dropout 0.2 as mentioned in the paper
    x = Flatten()(x) # Output: 1536

    # Output layer
    output = Dense(units = nb_classes, activation = "sigmoid")(x) # Output: 1000

    model = Model(init, output, name = "Inception-v4")

    return model

def load_model():

    inception_v4 = inception_v4()
    inception_v4.summary()

    dot_img_file = '/content/drive/MyDrive/Assignments/Assignment 3/InceptionV4.png'
    tf.keras.utils.plot_model(inception_v4, to_file=dot_img_file, show_shapes=True, dpi = 500)
    model = inception_v4

    return model 

def train_model(model, training_set, val_set, test_set):

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    history = model.fit(training_set,steps_per_epoch = 10,epochs = 200,validation_data = val_set,validation_steps = 5)

def plot_train_test_graph(history):

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
    pyplot.savefig('/content/drive/MyDrive/Assignments/Assignment 1/TrainTest_LossAccuracy_Graph_EfficientNetB0.png')
    pyplot.show()

def classification(model, test_set):

    test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)

    predictions = model.predict_generator(test_set, steps=test_steps_per_epoch)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_set.classes
    class_labels = list(test_set.class_indices.keys())

    return true_classes, predicted_classes, class_labels

def classification_metrics(true_classes, predicted_classes, class_labels):

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

    plt.savefig('/content/drive/MyDrive/Paper_Publications_Files/Paper_3/Confusion_Matrix_InceptionV4.png')
