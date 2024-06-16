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
import pytorch_ssim  # Import the SSIM library

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
    
import random
class ClearImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = glob(os.path.join(dataset, '*.bmp')) 
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]
        image = cv2.imread(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        return image

    def random_image(self):
        random_index = random.randint(0, len(self.dataset) - 1)
        random_image = self.dataset[random_index]
        random_image = cv2.imread(random_image, cv2.COLOR_BGR2RGB)
        if self.transform:
            random_image = self.transform(random_image)
        return random_image

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
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            BaseConv(3, 64, 3, 2, activation=nn.LeakyReLU(0.2), use_bn=False),  # Input: 3x256x256, Output: 64x128x128
            BaseConv(64, 128, 3, 2, activation=nn.LeakyReLU(0.2), use_bn=True),  # Output: 128x64x64
            BaseConv(128, 256, 3, 2, activation=nn.LeakyReLU(0.2), use_bn=True),  # Output: 256x32x32
            BaseConv(256, 512, 3, 2, activation=nn.LeakyReLU(0.2), use_bn=True),  # Output: 512x16x16
            BaseConv(512, 512, 3, 2, activation=nn.LeakyReLU(0.2), use_bn=True),  # Output: 512x8x8
            BaseConv(512, 512, 3, 2, activation=nn.LeakyReLU(0.2), use_bn=True),  # Output: 512x4x4
            BaseConv(512, 512, 3, 1, activation=nn.LeakyReLU(0.2), use_bn=True),  # Output: 512x4x4
            nn.Conv2d(512, 1, 4, 1, 0),  # Output: 1x1x1 (Real/Fake)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)
    
def Create_nets():
    generator = Generator()
    discriminator_clear = Discriminator()
    discriminator_blur = Discriminator()

    if torch.cuda.is_available():
        generator = generator.cuda()
        discriminator_clear = discriminator_clear.cuda()
        discriminator_blur = discriminator_blur.cuda()
        
    return generator, discriminator_clear, discriminator_blur

def psnr_loss(output, target):
    output_clipped = torch.clamp(output, min=0.0, max=1.0)
    mse_loss = nn.MSELoss()(output_clipped, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
    return psnr

def save_image(images, output_dir, epoch, tag):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        if(tag == 'mask'):
            img = img.squeeze(0).cpu().numpy()  # Remove channel dimension if it exists
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img, mode='L')
        elif(tag == 'output'or tag == 'clear' or tag == 'blur'):
            img = img.detach().cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy array
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    clear_image_dir = '/home/oscar/Desktop/dataset/SG_dataset/dataset/train_data/FCFB/FCC/FC/'
    clear_image_loader = ClearImageDataset(clear_image_dir, transform=transform)

    blur_image_dir = '/home/oscar/Desktop/dataset/SG_dataset/dataset/train_data/FCFB/FBB/FB'
    blur_image_loader = ClearImageDataset(blur_image_dir, transform=transform)

    generator, discriminator_clear, discriminator_blur = Create_nets()
    optimizer = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D_clear = optim.Adam(discriminator_clear.parameters(), lr=learning_rate)
    optimizer_D_blur = optim.Adam(discriminator_blur.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    min_loss = 100
    lambda_gp = 10

    for epoch in range(epochs):
        generator.train()
        discriminator_clear.train()
        discriminator_blur.train()
        epoch_loss_gen = 0
        epoch_loss_D_clear = 0
        epoch_loss_D_blur = 0

        for img, mask_img, gt_img in tqdm(dataloader):
            img, mask_img, gt_img = img.to(device), mask_img.to(device), gt_img.to(device)

            optimizer.zero_grad()
            output = generator(img, mask_img)
            clear_image = clear_image_loader.random_image().to(device)
            blur_image = blur_image_loader.random_image().to(device)

            clear_output = output * mask_img + clear_image * (1 - mask_img)
            blur_output = output * (1 - mask_img) + blur_image * mask_img

            valid = torch.ones(clear_output.size(0), 1, device=device)*0.9
            fake = torch.zeros(blur_output.size(0), 1, device=device)*0.1
            
            # Train discriminator_clear
            optimizer_D_clear.zero_grad()
            pred_clear = discriminator_clear(clear_output)
            loss_clear = criterion(pred_clear, valid)
            loss_clear.backward(retain_graph=True)
            optimizer_D_clear.step()
            
            # Train discriminator_blur
            optimizer_D_blur.zero_grad()
            pred_blur = discriminator_blur(blur_output)
            loss_blur = criterion(pred_blur, fake)
            loss_blur.backward(retain_graph=True)
            optimizer_D_blur.step()

            # Now update the generator
            optimizer.zero_grad()
            pred_clear = discriminator_clear(output * mask_img + blur_image * (1 - mask_img))
            pred_blur = discriminator_blur(output * (1 - mask_img) + clear_image * mask_img)
            loss_gen_clear = criterion(pred_clear, fake)
            loss_gen_blur = criterion(pred_blur, valid)
            loss_gen = loss_gen_clear + loss_gen_blur

            # Pass the generator's output back into the generator
            new_output = generator(output, mask_img)
            
            # Compute SSIM loss
            ssim_loss = 1 - pytorch_ssim.ssim(new_output, img)
            
            # Total generator loss
            total_gen_loss = loss_gen + ssim_loss
            total_gen_loss.backward()
            optimizer.step()

            epoch_loss_D_clear += loss_clear.item()
            epoch_loss_D_blur += loss_blur.item()
            epoch_loss_gen += total_gen_loss.item()

        avg_epoch_loss_D_clear = epoch_loss_D_clear / len(dataloader)
        avg_epoch_loss_D_blur = epoch_loss_D_blur / len(dataloader)
        avg_epoch_loss_gen = epoch_loss_gen / len(dataloader)
        total_loss = avg_epoch_loss_D_clear + avg_epoch_loss_D_blur + avg_epoch_loss_gen

        if(1):
            min_loss = total_loss
            result_path = './0528_result'

            save_image(img, result_path, epoch, 'input')
            save_image(mask_img, result_path, epoch, 'mask')
            save_image(output, result_path, epoch, 'output')
            save_image(gt_img, result_path, epoch, 'gt')
            save_image(clear_output, result_path, epoch, 'clear')
            save_image(blur_output, result_path, epoch, 'blur')

            save_path_G = os.path.join(model_save_path, f'generator_{epoch+1}.pth')
            save_path_D_clear = os.path.join(model_save_path, f'discriminator_clear_{epoch+1}.pth')
            save_path_D_blur = os.path.join(model_save_path, f'discriminator_blur_{epoch+1}.pth')
            torch.save(generator.state_dict(), save_path_G)
            torch.save(discriminator_clear.state_dict(), save_path_D_clear)
            torch.save(discriminator_blur.state_dict(), save_path_D_blur)
            generator.eval()
            discriminator_clear.eval()
            discriminator_blur.eval()

        print(f"Epoch [{epoch+1}/{epochs}], \
              D_clear Loss: {avg_epoch_loss_D_clear:.4f}, \
              D_blur Loss: {avg_epoch_loss_D_blur:.4f}, \
              Gen Loss: {avg_epoch_loss_gen:.4f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a simple image generator.')
    parser.add_argument('--image_dir', type=str, default='/home/oscar/Desktop/SG/results_0423/blurred_target')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
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
