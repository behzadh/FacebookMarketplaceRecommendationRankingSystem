from torchvision import transforms
from PIL import Image
import torch

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]) 
        ])

    def __call__(self, image):

        image = self.transform(image)
        image = image[None, :, :, :] # Add a dimension to the image
        print(image.size())
        return image
        
image_path = '/Users/behzad/AiCore/fb_raw_data/cleaned_images/' + '0a1d0925-d2aa-4e89-b9d3-ef56b834cfd9_resized.jpg'
new_input = Image.open(image_path)

dataset = ImageProcessor()
dataloader = torch.utils.data.DataLoader(dataset.__call__(new_input), batch_size=1)
