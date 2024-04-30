from explainable_ai.LIME import *
from explainable_ai.GRADCAM import *
from explainable_ai.Occlusion import *
from explainable_ai.SaliencyMap import *
from explainable_ai.SHAP_and_Smooth_Grad import *

if __name__ == "main":

    project_path = '/content/drive/MyDrive/Academic_Courses_and_ML_Projects/Paper_Publications_Files/Paper_3'
    image_url = project_path + '/Model_Training_Dataset/Train/Malignant/Folder_20_3_1_img4.jpg'
    ResNet60model = load_model(project_path + '/Ovarian_Image_classification_ResNet60.h5')

    # LIME
    run_LIME(image_url, ResNet60model)

    # GRAD CAM
    image = load_image(image_url)
    class_idx = 1 # class for which we need interpretation: benign or malignant
    run_GRADCAM(ResNet60model, class_idx, image)

    # Occlusion Analysis
    run_Occlusion(image_url, ResNet60model)

    # Saliency Map
    run_SaliencyMap(image_url, ResNet60model)

    # SHAP and Smooth Grad
    sample_image, explainer = run_SHAP(image_url, ResNet60model)
    run_Smooth_Grad(sample_image, explainer)





