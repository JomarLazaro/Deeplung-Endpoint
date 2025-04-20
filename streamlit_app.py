import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import base64

class ChestXRayModel:
    def __init__(self, model_path="deeplung-model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = ["NORMAL", "PNEUMONIA", "UNKNOWN", "TUBERCULOSIS"]

        self.transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

    def predict(self, image):
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = Image.fromarray(image).convert("RGB")

        image_tensor = self.transformations(image).to(self.device).unsqueeze(0)
        output = self.model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        pred_idx = output.argmax(dim=1).item()
        confidence = probabilities[0, pred_idx].item() * 100
        
        all_probs = {self.categories[i]: round(float(probabilities[0, i].item()) * 100, 2) 
                     for i in range(len(self.categories))}
        
        return {
            "prediction": self.categories[pred_idx],
            "confidence": round(confidence, 2),
            "probabilities": all_probs
        }

# Initialize once
model = ChestXRayModel()

# Streamlit UI
st.title("Chest X-Ray Disease Detector")

uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["png", "jpg", "jpeg"])
base64_input = st.text_area("Or paste base64 string (with or without prefix)")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    result = model.predict(uploaded_file.read())
    st.write(result)

elif base64_input:
    try:
        clean_base64 = base64_input.split("base64,")[-1]
        image_data = base64.b64decode(clean_base64)
        result = model.predict(image_data)
        st.image(Image.open(io.BytesIO(image_data)), caption="Decoded Image", use_column_width=True)
        st.write(result)
    except Exception as e:
        st.error(f"Invalid base64 input: {e}")
