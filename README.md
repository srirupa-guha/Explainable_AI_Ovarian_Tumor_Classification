# Explainable_AI_Ovarian_Tumor_Classification
This repository contains the codes for custom ResNet60 and other ILSVRC winning CNN architectures implemented on Ovarian tumor dataset for classification of tumors as benign or malignant. 

Below architectures are implemented:
- Custom ResNet60
- DenseNet121
- EfficientNetB0
- GoogLeNet
- InceptionResNet
- InceptionV4
- ResNet16
- ResNeXt
- VGG19

In addition to these, the below Explainable AI methods for interpretation of the classification output of ResNet60 on these images are also included in this repository:
- LIME
- Occlusion Analysis
- Saliency Map
- GRAD CAM
- SHAP and Smooth Grad

In this project, we have implemented and compared various state-of-the-art CNN architectures on a custom ovarian tumor dataset. The dataset comprises CT scanned images of ovarian tumors from the axial, saggital and coronal planes, belonging to the two classes: benign and malignant. Based on the comparison of the classification performance of these architectures on the CT scan dataset, the custom ResNet60 architecture was selected as the best performing model since the validation and test accuracy of the ResNet60 was higher on the ovarian tumor dataset as compared to the other models. Finally this ResNet60 architecture is carried forward for explainability.

As a next step, we implement the above mentioned Explainable AI techniques on the classification results of ResNet60 on the ovarian tumor dataset in order to interpret the classification. This interpretation helps to identify the important features or highlight the regions of interest which influence the final classification output - benign or malignant.

Steps to run the repository:

1. Clone the main branch of this repo: git clone https://github.com/srirupa-guha/Explainable_AI_Ovarian_Tumor_Classification.git
2. Create a conda environment with relevant packacges installed: <br>
    using pip: <br>
    pip install -r requirements.txt <br>
    using conda: <br>
    conda create --name <env_name> --file requirements.txt <br>
    Note: Each architecture has it's own requirements.txt file, for example, requirements_ResNet60.txt <br>
4. Python requirements: Python 3.8 or above
5. For training CNN architectures, run train.py
6. For CNN classification, run classification.py
7. For generating classification metrics, run metrics.py
8. For implementing explainable AI techniques, run interpret.py
