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
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'ResNet60'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)

    # DenseNet121
    base_model, preds = DenseNet()
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'DenseNet121'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)

    # EfficientNetB0
    model = EfficientNetB0()
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'EfficientNetB0'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)

    # GoogLeNet
    model = load_model_GoogleNet()
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'GoogLeNet'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)

    # InceptionResNet
    model = InceptionResNet()
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'InceptionResNet'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)

    # InceptionV4
    model = load_model_InceptionV4()
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'InceptionV4'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)

    # ResNet16
    model = load_model_ResNet16()
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'ResNet16'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)

    # ResNext
    model = load_model_ResNext()
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'ResNext'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)

    # VGG19
    model = load_model_VGG19()
    true_classes, predicted_classes, class_labels = classification(model, test_set)
    model_version = 'VGG19'
    classification_metrics(true_classes, predicted_classes, class_labels, model_version)