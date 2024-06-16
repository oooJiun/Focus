import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from my_datasets import Get_dataloader
from my_ssim import ssim

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob(os.path.join(image_dir, '*.png'))  # adjust the extension as needed
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(os.path.join("./source_map", self.image_paths[idx].split('/')[-1]), cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(os.path.join("/home/oscar/Desktop/dataset/DPDD/dd_dp_dataset_png/train_c/source", self.image_paths[idx].split('/')[-1]), cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
            mask_img = self.transform(mask_img)
            gt_img = self.transform(gt_img)

        return image, mask_img, gt_img
    
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
            BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True),
        )

        self.decoder = nn.Sequential(
            BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(256, 256, 1, 1, activation=nn.ReLU(), use_bn=True),
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
    
def Create_nets():
    generator = Generator()

    if torch.cuda.is_available():
        generator = generator.cuda()
        
    return generator

def psnr_loss(output, target):

    output_clipped = torch.clamp(output, min=0.0, max=1.0)
    mse_loss = nn.MSELoss()(output_clipped, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
    return psnr

def save_images(images, output_dir, epoch, tag):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        if(tag == 'mask'):
            img = img.squeeze(0).cpu().numpy()  # Remove channel dimension if it exists
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img, mode='L')
        elif(tag == 'output'):
            img = img.detach().cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy array
            # img = img.permute(1, 2, 0)
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img)
        else:
            img = img.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f'{i}_{tag}.png'))

def train_model(image_dir, epochs, batch_size, learning_rate, model_save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])

    dataset = ImageDataset(image_dir, transform)
    # dataloader = Get_dataloader(dataset, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Create_nets()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    min_loss = 100
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for img, mask_img, gt_img in tqdm(dataloader):
            img, mask_img, gt_img = img.to(device), mask_img.to(device), gt_img.to(device)
            optimizer.zero_grad()
            output = model(img, mask_img)
            # loss = psnr_loss(output, gt_img)
            loss = 1-ssim(output, gt_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        if(avg_epoch_loss<min_loss):
            min_loss = avg_epoch_loss
            save_path = os.path.join(model_save_path, str(epoch)+'.pth')
            torch.save(model.state_dict(), save_path)
            result_path = './0521_result'
            save_images(img, result_path, epoch, 'input')
            save_images(mask_img, result_path, epoch, 'mask')
            save_images(output, result_path, epoch, 'output')
            save_images(gt_img, result_path, epoch, 'gt')

        print(f"Epoch [{epoch+1}/{epochs}], PSNR Loss: {avg_epoch_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path)
            model.eval()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a simple image generator.')
    parser.add_argument('--image_dir', type=str, default='/home/oscar/Desktop/SG/results_0423/blurred_target')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='Learning rate for the optimizer.')
    parser.add_argument('--model_save_path', type=str, default='./saved_models', help='Path to save the trained model.')

    args = parser.parse_args()

    train_model(
        image_dir=args.image_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=args.model_save_path
    )
