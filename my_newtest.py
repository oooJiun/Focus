import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from my_ssim import ssim  # Assuming you have this module

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob(os.path.join(image_dir, '*.png'))  # Adjust the extension as needed
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(os.path.join("./source_map", self.image_paths[idx].split('/')[-1]), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
            mask_img = self.transform(mask_img)
        return image, mask_img

class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, gt_dir, mask2_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.png')))  # Adjust extension if needed
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))    # Adjust extension if needed
        assert len(self.image_paths) == len(self.mask_paths), \
            "The number of images, masks, and ground truth images must be the same"
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            mask_img = self.transform(mask_img)

        return image, mask_img

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)
        return input

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            BaseConv(4, 64, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(256, 256, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True),
            # BaseConv(512, 512, 1, 1, activation=nn.ReLU(), use_bn=True),
            # BaseConv(512, 512, 1, 1, activation=nn.ReLU(), use_bn=True),
        )

        self.decoder = nn.Sequential(
            BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True),
            # BaseConv(256, 256, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(128, 64, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(32, 3, 3, 1, activation=None, use_bn=False),
        )

    def forward(self, real_image, mask):
        input_combined = torch.cat([real_image, mask], dim=1)
        encoder = self.encoder(input_combined)
        decoder = self.decoder(encoder)

        return decoder

def load_model(model_path, device):
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def infer_image(image_path, mask_path, model, device):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    print(image_path)
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image = transform(image).unsqueeze(0).to(device)
    mask_img = transform(mask_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image, mask_img)
    
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).astype(np.uint8)
    output_img = Image.fromarray(output)
    
    return output_img

def infer_folder(image_dir, mask_dir, model, device, output_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob(os.path.join(image_dir, '*.png')))  # Adjust extension if necessary
    mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))
    for image_path, mask_path in zip(image_paths, mask_paths):
        img_name = os.path.basename(image_path)
        # mask_path = os.path.join(mask_dir, img_name)
        
        if not os.path.exists(mask_path):
            print(f"Mask for image {img_name} not found. Skipping...")
            continue
        
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = transform(image).to(device)
        mask_img = transform(mask_img).to(device)

        with torch.no_grad():
            output_img = model(image, mask_img)

        
        
        # output_img = infer_image(image_path, mask_path, model, device)
        # img = output_img.squeeze(0)
        img = output_img.detach().cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy array
        # img = img.permute(1, 2, 0)
        img = (img * 255).astype(np.uint8)  # Convert to uint8 type
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, img_name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Inference with the trained image generator model.')
    parser.add_argument('--image_dir', type=str, help='Path to the input image.')
    parser.add_argument('--mask_dir', type=str, help='Path to the binary mask image.')
    parser.add_argument('--model_path', type=str, default = 'saved_models/generator_1000.pth', help='Path to the trained model.')
    parser.add_argument('--output_dir', type=str, default = '0611_result/2/test', help='Path to save the output image.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_path, device)
    infer_folder(args.image_dir, args.mask_dir, model, device, args.output_dir)
    print(f"Output images saved to {args.output_dir}")


'''
python3 my_newtest.py --image_dir /home/oscar/Desktop/dataset/DPDD/dd_dp_dataset_png/train_c/source --mask_dir /home/oscar/Desktop/SG/target_map
'''