import torch, torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
from pathlib import Path

# ---- paths ----
ckpt_path = "my_ecg_model_vgg16.pth"

img_size  = 225
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- rebuild model exactly like training ----
def build_model(nc):
    m = models.vgg16(weights=None)
    in_f = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.6),
        nn.BatchNorm1d(in_f),
        nn.Linear(in_f, nc)
    )
    return m

ckpt = torch.load(ckpt_path, map_location=device)
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v:k for k,v in class_to_idx.items()}

model = build_model(len(class_to_idx))
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# ---- transforms ----
tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])



# OPTION 2: single file path (infer expected from folder name or pass manually)
def predict_path(path, expected=None):
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(1).item()
        conf = torch.softmax(logits, 1)[0, pred_idx].item()
    pred_cls = idx_to_class[pred_idx]
    if expected is None:
        expected = Path(path).parent.name  # ImageFolder-style folder = label
    print(f"Predicted: {pred_cls} ({conf:.4f}) | Actual: {expected}")
    return img, pred_cls, expected,conf

