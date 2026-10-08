import streamlit as st
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

import cv2
import numpy as np


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🫁",
    layout="centered"
)


# =========================================================
# 2. TITLE
# =========================================================

st.markdown(
    "<h1 style='text-align:center;'>🫁 Chest X-Ray Pneumonia Detector</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "AI-powered chest X-ray classification using ResNet18"
    "</p>",
    unsafe_allow_html=True
)


# =========================================================
# 3. INFORMATION
# =========================================================

st.info(
    "Upload a chest X-ray image. The trained ResNet18 model "
    "will classify it as NORMAL or PNEUMONIA and generate "
    "a Grad-CAM visualization showing regions that influenced "
    "the prediction."
)


# =========================================================
# 4. DEVICE
# =========================================================

device = torch.device("cpu")


# =========================================================
# 5. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = models.resnet18(
        weights=None
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "models/pneumonia_resnet18_improved.pth",
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model


model = load_model()


# =========================================================
# 6. CLASS NAMES
# =========================================================

classes = [
    "NORMAL",
    "PNEUMONIA"
]


# =========================================================
# 7. IMAGE TRANSFORMATION
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 8. UPLOAD IMAGE
# =========================================================

st.subheader("📤 Upload Chest X-Ray")

uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# 9. GRAD-CAM FUNCTION
# =========================================================

def generate_gradcam(
    model,
    input_tensor,
    original_image,
    predicted_class
):

    activations = []
    gradients = []


    # -----------------------------
    # Forward hook
    # -----------------------------

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output
        )


    # -----------------------------
    # Backward hook
    # -----------------------------

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0]
        )


    # -----------------------------
    # Target layer
    # -----------------------------

    target_layer = model.layer4[-1]


    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    # -----------------------------
    # Forward pass
    # -----------------------------

    model.zero_grad()

    output = model(
        input_tensor
    )


    # -----------------------------
    # Backward pass
    # -----------------------------

    score = output[
        0,
        predicted_class
    ]

    score.backward()


    # -----------------------------
    # Remove hooks
    # -----------------------------

    forward_handle.remove()

    backward_handle.remove()


    # -----------------------------
    # Get activation
    # -----------------------------

    activation = (
        activations[0][0]
        .detach()
    )

    gradient = (
        gradients[0][0]
        .detach()
    )


    # -----------------------------
    # Calculate weights
    # -----------------------------

    weights = gradient.mean(
        dim=(1, 2)
    )


    # -----------------------------
    # Generate CAM
    # -----------------------------

    cam = torch.zeros(
        activation.shape[1:],
        dtype=torch.float32
    )


    for i in range(
        activation.shape[0]
    ):

        cam += (
            weights[i]
            * activation[i]
        )


    cam = torch.relu(
        cam
    )


    cam = cam.cpu().numpy()


    # -----------------------------
    # Resize CAM
    # -----------------------------

    cam = cv2.resize(
        cam,
        (224, 224)
    )


    # -----------------------------
    # Normalize
    # -----------------------------

    cam = (
        cam - cam.min()
    ) / (
        cam.max()
        - cam.min()
        + 1e-8
    )


    # -----------------------------
    # Heatmap
    # -----------------------------

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # -----------------------------
    # Original image
    # -----------------------------

    original = np.array(
        original_image
    )

    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    original = cv2.resize(
        original,
        (224, 224)
    )


    # -----------------------------
    # Overlay
    # -----------------------------

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )


    # Convert BGR → RGB
    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    return overlay


# =========================================================
# 10. PROCESS UPLOADED IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -----------------------------
    # Display uploaded image
    # -----------------------------

    st.subheader(
        "🩻 Uploaded X-Ray"
    )

    st.image(
        image,
        use_container_width=True
    )


    # -----------------------------
    # Convert image to tensor
    # -----------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0)

    input_tensor = input_tensor.to(
        device
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[
                0,
                predicted_class
            ].item()
            * 100
        )


    prediction = classes[
        predicted_class
    ]


    # =====================================================
    # RESULT
    # =====================================================

    st.subheader(
        "🔍 Prediction Result"
    )


    if prediction == "PNEUMONIA":

        st.error(
            f"🫁 Prediction: {prediction}"
        )

    else:

        st.success(
            f"✅ Prediction: {prediction}"
        )


    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.subheader(
        "📊 Class Probabilities"
    )


    normal_probability = (
        probabilities[0, 0].item()
        * 100
    )

    pneumonia_probability = (
        probabilities[0, 1].item()
        * 100
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "NORMAL",
            f"{normal_probability:.2f}%"
        )

        st.progress(
            int(normal_probability)
        )


    with col2:

        st.metric(
            "PNEUMONIA",
            f"{pneumonia_probability:.2f}%"
        )

        st.progress(
            int(pneumonia_probability)
        )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.subheader(
        "🔥 Grad-CAM Explanation"
    )

    st.write(
        "The heatmap shows image regions that "
        "influenced the model's prediction."
    )


    # Grad-CAM needs gradients
    input_tensor_grad = input_tensor.clone()
    input_tensor_grad.requires_grad_(True)


    gradcam_image = generate_gradcam(
        model,
        input_tensor_grad,
        image,
        predicted_class
    )


    st.image(
        gradcam_image,
        caption="Grad-CAM Heatmap",
        use_container_width=True
    )


# =========================================================
# 11. MODEL PERFORMANCE
# =========================================================

st.markdown("---")

st.subheader(
    "📈 Model Performance"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Test Accuracy",
        "89.74%"
    )


with col2:

    st.metric(
        "Normal Accuracy",
        "74.36%"
    )


with col3:

    st.metric(
        "Pneumonia Accuracy",
        "98.97%"
    )


st.write(
    "**Pneumonia Precision:** 86.55%"
)

st.write(
    "**Pneumonia Recall:** 98.97%"
)

st.write(
    "**Pneumonia F1 Score:** 92.34%"
)


# =========================================================
# 12. MODEL INFORMATION
# =========================================================

st.markdown("---")

st.subheader(
    "ℹ️ Model Information"
)

st.write(
    "**Model:** ResNet18"
)

st.write(
    "**Dataset:** Chest X-Ray Pneumonia"
)

st.write(
    "**Classes:** NORMAL, PNEUMONIA"
)

st.write(
    "**Device:** CPU"
)


# =========================================================
# 13. WARNING
# =========================================================

st.warning(
    "⚠️ This is an educational AI project and is not "
    "a medical diagnostic tool. Predictions should not "
    "replace evaluation by a qualified healthcare professional."
)


# =========================================================
# 14. FOOTER
# =========================================================

st.markdown(
    "<p style='text-align:center;'>"
    "Built with Python • PyTorch • ResNet18 • Streamlit • Grad-CAM"
    "</p>",
    unsafe_allow_html=True
)