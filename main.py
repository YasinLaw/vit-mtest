import requests
import torch.cuda
from PIL import Image
from transformers import (
    AutoImageProcessor,
    ViTModel,
    ViTImageProcessor,
    ResNetForImageClassification,
    AutoFeatureExtractor,
)

batch_size = 512
curr = 6


model_list = [
    "google/vit-base-patch16-224-in21k",
    "google/vit-base-patch32-224-in21k",
    "google/vit-huge-patch14-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "google/vit-large-patch32-224-in21k",
    "microsoft/resnet-50",
    "microsoft/resnet-18",
]

if curr >= len(model_list):
    exit(-1)

torch.set_default_device("cuda")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

images = []

processor = None
model = None

if "resnet" not in model_list[curr]:
    processor = ViTImageProcessor.from_pretrained(model_list[curr])
    model = ViTModel.from_pretrained(model_list[curr])
elif "50" not in model_list[curr]:
    processor = AutoImageProcessor.from_pretrained(model_list[curr])
    model = ResNetForImageClassification.from_pretrained(model_list[curr])
elif "18" not in model_list[curr]:
    processor = AutoFeatureExtractor.from_pretrained(model_list[curr])
    model = ResNetForImageClassification.from_pretrained(model_list[curr])

for i in range(0, batch_size):
    images.append(image)
inputs = processor(images=images, return_tensors="pt")
outputs = model(**inputs)
torch.cuda.empty_cache()
with open(
    model_list[curr].replace("/", "-") + "-" + str(batch_size) + ".txt", "w"
) as f:
    print(torch.cuda.memory_summary())
    f.write(model_list[curr] + "\n" + torch.cuda.memory_summary())
