from torchvision.transforms import Compose, Grayscale, Resize, ToTensor


def get_inference_transform(image_size: int, in_channels: int):
    return Compose([
        Grayscale(num_output_channels=in_channels),
        Resize((image_size, image_size)),
        ToTensor(),
    ])
