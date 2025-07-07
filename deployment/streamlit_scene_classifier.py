# streamlit_scene_classifier.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import time
from model_definitions import HybridEfficientCNN_MIT67, DenseNetLikeSceneClassifier, MiniEfficientNet, FosNet  # make sure this file exists
from class_names import label_names  # list of 67 scene class names

# Setup
st.set_page_config(page_title="Scene Classifier", page_icon="🧠", layout="centered")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
NUM_CLASSES = 67

# Available models
MODEL_OPTIONS = {
    "HybridEfficientCNN": ("project_HybridCNNSE_sarayusi_vishalim.pth", HybridEfficientCNN_MIT67),
    "DenseNetLike": ("project_DenseNet_sarayusi_vishalim.pth", DenseNetLikeSceneClassifier),
    "MiniEfficientNet": ("project_sarayusi_vishalim_efficientnet_student.pth", MiniEfficientNet),
    "FosNet": ("project_sarayusi_vishalim_fosnet_student.pth", FosNet)
}

@st.cache_resource
def load_model(model_name):
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

def predict(image_tensor, model):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        return label_names[predicted.item()], confidence.item()

def main():
    st.title("Indoor Scene Classifier")
    st.markdown("Upload a scene image to classify it into one of 67 MIT indoor scene categories.")

    st.sidebar.header("Choose Your Model")
    selected_model = st.sidebar.selectbox("Select Model", list(MODEL_OPTIONS.keys()))

    st.sidebar.write("Model selected:", selected_model)
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict"):
            with st.spinner("Loading model and making prediction..."):
                model = load_model(selected_model)
                transform = get_transforms()
                input_tensor = transform(image).unsqueeze(0)

                label, confidence = predict(input_tensor, model)
                time.sleep(1.0)

                st.success(f" Predicted Scene: **{label}**")
                st.info(f"Confidence Score: **{confidence:.2%}**")

if __name__ == "__main__":
    main()
