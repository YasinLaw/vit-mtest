import torch.cuda
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from transformers import (
    AutoImageProcessor,
    ViTModel,
    ViTImageProcessor,
    ResNetForImageClassification,
    AutoFeatureExtractor,
)

b_size = 32
curr = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds = datasets.ImageFolder(
    root="/home/luo/datasets/celeba", transform=transforms.ToTensor()
)

dl = DataLoader(ds, batch_size=b_size)

model_list = [
    "google/vit-base-patch16-224-in21k",  # 0
    "google/vit-base-patch32-224-in21k",  # 1
    "google/vit-huge-patch14-224-in21k",  # 2
    "google/vit-large-patch16-224-in21k",  # 3
    "google/vit-large-patch32-224-in21k",  # 4
    "microsoft/resnet-50",  # 5
    "microsoft/resnet-18",  # 6
]

if curr >= len(model_list):
    exit(-1)


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# image = len(list(image.getdata()))

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

model = model.to(device)

for i, data in enumerate(dl, 0):
    real_data = data[0].to(device)
    inputs = processor(images=real_data, return_tensors="pt").to(device)
    outputs = model(**inputs)
    with open(
        model_list[curr].replace("/", "-") + "-" + str(b_size) + ".txt", "w"
    ) as f:
        print(torch.cuda.memory_summary())
        f.write(model_list[curr] + "\n" + torch.cuda.memory_summary())
    break

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# torch.cuda.empty_cache()
# with open(model_list[curr].replace("/", "-") + "-" + str(b_size) + ".txt", "w") as f:
#     print(torch.cuda.memory_summary())
#     f.write(model_list[curr] + "\n" + torch.cuda.memory_summary())
