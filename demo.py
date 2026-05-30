import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import pickle

# Noms des pathologies
label_names = ['atelectasis', 'cardiomegaly', 'effusion', 'infiltration',
               'mass', 'nodule', 'pneumonia', 'pneumothorax',
               'consolidation', 'edema', 'emphysema', 'fibrosis', 'pleural', 'hernia']

# Architectures
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

class MultimodalFusion(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128*8*8, 256), nn.ReLU()
        )
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_fc = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 14)
        )
    def forward(self, img, text):
        img_feat = self.cnn(img)
        txt_feat = self.embed(text).mean(dim=1)
        txt_feat = self.text_fc(txt_feat)
        return self.classifier(torch.cat([img_feat, txt_feat], dim=1))

# Chargement des modeles
@st.cache_resource
def load_models():
    cnn = CNNScratch()
    cnn.load_state_dict(torch.load('models/cnn_scratch.pth', map_location='cpu'))
    cnn.eval()

    ae = Autoencoder()
    ae.load_state_dict(torch.load('models/autoencoder.pth', map_location='cpu'))
    ae.eval()

    with open('models/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab) + 1
    mm = MultimodalFusion(vocab_size)
    mm.load_state_dict(torch.load('models/model_multimodal.pth', map_location='cpu'))
    mm.eval()

    return cnn, ae, mm, vocab

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

def text_to_tensor(text, vocab, max_len=50):
    words = text.lower().split()[:max_len]
    ids = [vocab.get(w, 0) for w in words]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

# Interface Streamlit
st.title("Systeme d'aide au tri radiologique")
st.markdown("Chargez une radiographie thoracique pour obtenir une analyse automatique.")

uploaded_file = st.file_uploader("Charger une radiographie", type=["png", "jpg", "jpeg"])
report_text = st.text_area("Compte-rendu radiologique (optionnel)",
                            placeholder="Ex: The cardiac silhouette is normal. No pleural effusion detected...")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Radio chargee", width=400)

    cnn, ae, mm, vocab = load_models()
    img_tensor = transform(image).unsqueeze(0)

    # Predictions CNN
    st.subheader("Predictions des pathologies (CNN)")
    with torch.no_grad():
        outputs = torch.sigmoid(cnn(img_tensor))
    probs = outputs.squeeze().numpy()
    for name, prob in zip(label_names, probs):
        st.progress(float(prob), text=f"{name} : {prob:.2%}")

    # Score d'anomalie
    st.subheader("Score d'anomalie")
    with torch.no_grad():
        reconstructed = ae(img_tensor)
    error = ((img_tensor - reconstructed) ** 2).mean().item()
    threshold = 0.0081
    st.metric("Erreur de reconstruction", f"{error:.4f}")
    if error > threshold:
        st.error(f"Image atypique detectee ! (seuil = {threshold})")
    else:
        st.success(f"Image normale (seuil = {threshold})")

    # Predictions multimodales
    if report_text.strip():
        st.subheader("Predictions multimodales (image + texte)")
        text_tensor = text_to_tensor(report_text, vocab)
        with torch.no_grad():
            mm_outputs = torch.sigmoid(mm(img_tensor, text_tensor))
        mm_probs = mm_outputs.squeeze().numpy()
        for name, prob in zip(label_names, mm_probs):
            st.progress(float(prob), text=f"{name} : {prob:.2%}")
    else:
        st.info("Ajoutez un compte-rendu radiologique pour obtenir les predictions multimodales.")