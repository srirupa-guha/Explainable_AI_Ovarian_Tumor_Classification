from cnn_models.Custom_ResNet60.ResNet60 import *
from cnn_models.DenseNet121.DenseNet121 import *
from cnn_models.EfficientNetB0.EfficientNetB0 import *
from cnn_models.GoogLeNet.GoogLeNet import *
from cnn_models.InceptionResNet.InceptionResNet import *
from cnn_models.InceptionV4.InceptionV4 import *
from cnn_models.ResNet16.ResNet16 import *
from cnn_models.ResNeXt.ResNeXt import *
from cnn_models.VGG19.VGG19 import *

if __name__ == "main":

    training_set, val_set, test_set = load_data()

    # Custom ResNet60
    model = load_model_ResNet60()
    history = train_model_ResNet60(model, training_set, val_set)
    plot_train_test_graph(history, 'ResNet60')

    # DenseNet121
    base_model, preds = DenseNet()
    model = load_model_DenseNet121(base_model, preds)
    history = train_model_DenseNet121(model, training_set, val_set, test_set)
    plot_train_test_graph(history, 'DenseNet121')

    # EfficientNetB0
    model = EfficientNetB0()
    history = train_model(model, training_set, val_set, test_set)
    plot_train_test_graph(history, 'EfficientNetB0')

    # GoogLeNet
    model = load_model_GoogleNet()
    history = train_model_GoogleNet(model, training_set, val_set)
    plot_train_test_graph(history, 'GoogLeNet')

    # InceptionResNet
    model = InceptionResNet()
    history = train_model(model, training_set, val_set, test_set)
    plot_train_test_graph(history, 'InceptionResNet')

    # InceptionV4
    model = load_model_InceptionV4()
    history = train_model_InceptionV4(model, training_set, val_set, test_set)
    plot_train_test_graph(history, 'InceptionResNet')

    # ResNet16
    model = load_model_ResNet16()
    history = train_model_ResNet16(model, training_set, val_set, test_set)
    plot_train_test_graph(history, 'ResNet16')

    # ResNext
    model = load_model_ResNext()
    history = train_model(model, training_set, val_set, test_set)
    plot_train_test_graph(history, 'ResNext')

    # VGG19
    model = load_model_VGG19()
    history = train_model_VGG19(model, training_set, val_set, test_set)
    plot_train_test_graph(history, 'VGG19')
    




