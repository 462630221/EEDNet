import time
import torch
from torchvision import models, transforms
from PIL import Image

from visualize import get_color_pallete
from EEDNet import EEDNet

def Seg_demo():
    print("---Segmentation Demo---")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
                                    transforms.Resize((1024, 2048)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img = Image.open("./1.png").convert('RGB')
    input = transform(img).unsqueeze(0).to(device)

    model = EEDNet(num_classes=19)
    model.load_state_dict(torch.load("./best_model.pth",map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    output = model(input)[0]
    pred = torch.argmax(output[0], 0).squeeze(0).cpu().data.numpy()

    mask = get_color_pallete(pred, dataset='citys')
    #mask = mask.resize((2048, 1024))
    mask.save("./out.png")
    mask.show()

if __name__ == "__main__":
    Seg_demo()











