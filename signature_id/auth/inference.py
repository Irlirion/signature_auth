import torch
from torchvision import transforms as T
from pytorch_metric_learning.utils.inference import InferenceModel

from signature_id.auth.model import get_model_resnet18


def crop_space(array, white=1):
    row = []
    for i in range(array.shape[1]):
        row.append((array[:, i, :] < white).any())
    array = array[:, row]
    column = []
    for i in range(array.shape[2]):
        column.append((array[:, :, i] < white).any())
    return array[:, :, column]

transform = T.Compose(
    [
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        crop_space,
        T.Resize((128, 128)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def get_inference_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trunk, embedder = get_model_resnet18(device)
    trunk.load_state_dict(torch.load("signature_id/weights/trunk_best1.pth", map_location=device))
    embedder.load_state_dict(torch.load("signature_id/weights/embedder_best1.pth", map_location=device))

    inference_model = InferenceModel(trunk=trunk, embedder=embedder, data_device=device)

    return inference_model
