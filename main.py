import torch.cuda
from transformers import (
    AutoImageProcessor,
    ViTModel,
    ViTImageProcessor,
    ResNetForImageClassification,
    AutoFeatureExtractor,
)
from PIL import Image
import requests

model_list = [
    "google/vit-base-patch16-224-in21k",
    "google/vit-base-patch32-224-in21k",
    "google/vit-huge-patch14-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "google/vit-large-patch32-224-in21k",
    "microsoft/resnet-50",
    "microsoft/resnet-18",
]
curr = 6

torch.set_default_device("cuda")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = None
model = None

if not model_list[curr].find("resnet"):
    processor = ViTImageProcessor.from_pretrained(model_list[curr])
    model = ViTModel.from_pretrained(model_list[curr])
elif model_list[curr].find("50"):
    processor = AutoImageProcessor.from_pretrained(model_list[curr])
    model = ResNetForImageClassification.from_pretrained(model_list[curr])
elif model_list[curr].find("18"):
    processor = AutoFeatureExtractor.from_pretrained(model_list[curr])
    model = ResNetForImageClassification.from_pretrained(model_list[curr])

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

with open(model_list[curr].replace("/", "-") + ".txt", "w") as f:
    print(torch.cuda.memory_summary())
    f.write(model_list[curr] + "\n" + torch.cuda.memory_summary())
