from torchvision.transforms import Compose, Grayscale, Resize, ToTensor, RandomApply, GaussianBlur


def get_inference_transform(image_size: int, in_channels: int):
    return Compose([
        Grayscale(num_output_channels=in_channels),
        Resize((image_size, image_size)),
        ToTensor(),
    ])


def get_training_transform(image_size: int, in_channels: int):
    return Compose([
        Grayscale(num_output_channels=in_channels),
        Resize((image_size, image_size)),
        RandomApply(
            transforms=[GaussianBlur(kernel_size=3)],
            p=0.1,
        ),
        ToTensor(),
    ])
