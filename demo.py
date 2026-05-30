import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ── Noms des pathologies ─────────────────────────────────
label_names = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration',
               'mass', 'nodule', 'pneumonia', 'pneumothorax',
               'consolidation', 'edema', 'emphysema', 'fibrosis', 'pleural', 'hernia']

# ── Architecture CNN ─────────────────────────────────────
class CNNScratch(nn.Module):
    def __init__(self, num_classes=14):
        super(CNNScratch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── Architecture Autoencoder ─────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2), nn.Tanh()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Chargement des modèles ───────────────────────────────
@st.cache_resource
def load_models():
    cnn = CNNScratch()
    cnn.load_state_dict(torch.load('models/cnn_scratch.pth', map_location='cpu'))
    cnn.eval()
    
    ae = Autoencoder()
    ae.load_state_dict(torch.load('models/autoencoder.pth', map_location='cpu'))
    ae.eval()
    
    return cnn, ae

# ── Transform ────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# ── Interface Streamlit ──────────────────────────────────
st.title("Système d'aide au tri radiologique")
st.markdown("Chargez une radiographie thoracique pour obtenir une analyse automatique.")

uploaded_file = st.file_uploader("Charger une radiographie", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Radio chargée", use_column_width=True)
    
    cnn, ae = load_models()
    img_tensor = transform(image).unsqueeze(0)
    
    # ── Prédictions supervisées ──────────────────────────
    st.subheader("Prédictions des pathologies")
    with torch.no_grad():
        outputs = torch.sigmoid(cnn(img_tensor))
    
    probs = outputs.squeeze().numpy()
    for name, prob in zip(label_names, probs):
        st.progress(float(prob), text=f"{name} : {prob:.2%}")
    
    # ── Score d'anomalie ─────────────────────────────────
    st.subheader("Score d'anomalie")
    with torch.no_grad():
        reconstructed = ae(img_tensor)
    
    error = ((img_tensor - reconstructed) ** 2).mean().item()
    threshold = 0.0081
    
    st.metric("Erreur de reconstruction", f"{error:.4f}")
    if error > threshold:
        st.error(f"Image atypique détectée ! (seuil = {threshold})")
    else:
        st.success(f"Image normale (seuil = {threshold})")