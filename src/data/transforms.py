from torchvision.transforms import (
    Compose,
    Grayscale,
    Resize,
    ToTensor,
    RandomApply,
    GaussianBlur,
    Normalize
)


def get_inference_transform(image_size: int, in_channels: int):
    return Compose([
        Grayscale(num_output_channels=in_channels),
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,)),
    ])


def get_training_transform(image_size: int, in_channels: int):
    return Compose([
        Grayscale(num_output_channels=in_channels),
        Resize((image_size, image_size)),
        RandomApply(
            transforms=[GaussianBlur(kernel_size=3)],
            p=0.25,
        ),
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,)),
    ])
