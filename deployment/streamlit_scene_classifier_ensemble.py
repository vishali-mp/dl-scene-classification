# # streamlit_scene_classifier_ensemble.py
# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# import os
# import time
# from model_definitions import HybridEfficientCNN_MIT67, DenseNetLikeSceneClassifier, MiniEfficientNet, FosNet
# from class_names import label_names

# # Setup
# st.set_page_config(page_title="Indoor Scene Classifier (Ensemble)", page_icon="📸", layout="centered")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NUM_CLASSES = 67

# MODEL_OPTIONS = {
#     "HybridEfficientCNN": ("project_HybridCNNSE_sarayusi_vishalim.pth", HybridEfficientCNN_MIT67),
#     "DenseNetLike": ("project_DenseNet_sarayusi_vishalim.pth", DenseNetLikeSceneClassifier),
#     "MiniEfficientNet": ("project_sarayusi_vishalim_efficientnet_student.pth", MiniEfficientNet),
#     "FosNet": ("project_sarayusi_vishalim_fosnet_student.pth", FosNet)
# }

# ENSEMBLE_MODEL_OPTIONS = {
#     "MiniEfficientNet": ("project_sarayusi_vishalim_efficientnet_student.pth", MiniEfficientNet),
#     "FosNet": ("project_sarayusi_vishalim_fosnet_student.pth", FosNet)
# }

# @st.cache_resource
# def load_all_models():
#     models = []
#     for name, (weight_path, model_class) in ENSEMBLE_MODEL_OPTIONS.items():
#         model = model_class(num_classes=NUM_CLASSES).to(device)
#         model.load_state_dict(torch.load(weight_path, map_location=device))
#         model.eval()
#         models.append(model)
#     return models

# @st.cache_data
# def get_transforms():
#     return transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

# def ensemble_predict(image_tensor, models):
#     with torch.no_grad():
#         outputs = []
#         for model in models:
#             out = model(image_tensor.to(device))
#             probs = torch.softmax(out, dim=1)
#             outputs.append(probs)

#         avg_probs = torch.mean(torch.stack(outputs), dim=0)
#         confidence, predicted = torch.max(avg_probs, dim=1)
#         return label_names[predicted.item()], confidence.item()

# def main():
#     st.title("Indoor Scene Classifier (Ensemble Mode)")
#     st.markdown("Upload a scene image to classify it into one of 67 MIT indoor scene categories.")

#     use_ensemble = st.sidebar.checkbox("Use Ensemble of All Models", value=True)

#     if not use_ensemble:
#         st.sidebar.header("Choose a Single Model")
#         selected_model = st.sidebar.selectbox("Select Model", list(MODEL_OPTIONS.keys()))

#     uploaded_file = st.file_uploader("\U0001F4E4 Upload an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         if st.button("\U0001F50D Predict"):
#             with st.spinner("Loading model(s) and making prediction..."):
#                 transform = get_transforms()
#                 input_tensor = transform(image).unsqueeze(0)

#                 if use_ensemble:
#                     models = load_all_models()
#                     label, confidence = ensemble_predict(input_tensor, models)
#                 else:
#                     model = load_model(selected_model)
#                     label, confidence = predict(input_tensor, model)

#                 time.sleep(1.0)
#                 st.success(f"Predicted Scene: **{label}**")
#                 st.info(f"Confidence Score: **{confidence:.2%}**")

# def load_model(model_name):
#     weight_path, model_class = MODEL_OPTIONS[model_name]
#     model = model_class(num_classes=NUM_CLASSES).to(device)
#     model.load_state_dict(torch.load(weight_path, map_location=device))
#     model.eval()
#     return model

# def predict(image_tensor, model):
#     with torch.no_grad():
#         outputs = model(image_tensor.to(device))
#         probs = torch.softmax(outputs, dim=1)
#         confidence, predicted = torch.max(probs, 1)
#         return label_names[predicted.item()], confidence.item()

# if __name__ == "__main__":
#     main()

# streamlit_scene_classifier_ensemble.py
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import time
from model_definitions import HybridEfficientCNN_MIT67, DenseNetLikeSceneClassifier, MiniEfficientNet, FosNet
from class_names import label_names

# Setup
st.set_page_config(page_title="Indoor Scene Classifier", page_icon="📸", layout="centered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 67

# Model paths and classes
MODEL_OPTIONS = {
    "HybridEfficientCNN": ("project_HybridCNNSE_sarayusi_vishalim.pth", HybridEfficientCNN_MIT67),
    "DenseNetLike": ("project_DenseNet_sarayusi_vishalim.pth", DenseNetLikeSceneClassifier),
    "MiniEfficientNet": ("project_sarayusi_vishalim_efficientnet_student.pth", MiniEfficientNet),
    "FosNet": ("project_sarayusi_vishalim_fosnet_student.pth", FosNet),
    "Ensemble (MiniEfficientNet + FosNet)": None  # Special handling
}

@st.cache_resource
def load_model(model_name):
    if model_name == "Ensemble (MiniEfficientNet + FosNet)":
        models = []
        for name in ["MiniEfficientNet", "FosNet"]:
            weight_path, model_class = MODEL_OPTIONS[name]
            model = model_class(num_classes=NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model.eval()
            models.append(model)
        return models  # list of models
    else:
        weight_path, model_class = MODEL_OPTIONS[model_name]
        model = model_class(num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        return model

@st.cache_data
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# def predict(image_tensor, model_or_models):
#     with torch.no_grad():
#         image_tensor = image_tensor.to(device)
#         if isinstance(model_or_models, list):  # Ensemble
#             outputs = []
#             for model in model_or_models:
#                 out = model(image_tensor)
#                 probs = torch.softmax(out, dim=1)
#                 outputs.append(probs)
#             avg_probs = torch.mean(torch.stack(outputs), dim=0)
#         else:  # Single model
#             output = model_or_models(image_tensor)
#             avg_probs = torch.softmax(output, dim=1)

#         confidence, predicted = torch.max(avg_probs, 1)
#         return label_names[predicted.item()], confidence.item()
def predict(image_tensor, model_or_models):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        if isinstance(model_or_models, list):  # Ensemble
            # Assign custom weights for the ensemble models
            weights = [0.6, 0.4]  # [MiniEfficientNet, FosNet]
            outputs = []
            for model, weight in zip(model_or_models, weights):
                out = model(image_tensor)
                probs = torch.softmax(out, dim=1)
                outputs.append(probs * weight)
            # Weighted sum of probabilities
            weighted_probs = torch.sum(torch.stack(outputs), dim=0)
            avg_probs = weighted_probs / sum(weights)  # normalize just in case
        else:
            output = model_or_models(image_tensor)
            avg_probs = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(avg_probs, 1)
        return label_names[predicted.item()], confidence.item()

def main():
    st.title("Indoor Scene Classifier")
    st.markdown("Upload an indoor scene image to classify it into one of 67 categories.")

    selected_model = st.sidebar.selectbox("Select Model", list(MODEL_OPTIONS.keys()))

    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict"):
            with st.spinner("Loading model(s) and making prediction..."):
                model_or_models = load_model(selected_model)
                transform = get_transforms()
                input_tensor = transform(image).unsqueeze(0)

                label, confidence = predict(input_tensor, model_or_models)
                time.sleep(1.0)

                st.success(f"**Predicted Scene**: {label}")
                st.info(f"**Confidence Score**: {confidence:.2%}")

if __name__ == "__main__":
    main()
