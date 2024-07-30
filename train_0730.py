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
import random
from LaKDNet import LaKDNet
from models import Generator_blur, NLayerDiscriminator
from NcolDran import MultiScalePatchDiscriminator
  
class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, clear_dir, blur_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.bmp'))) + \
                           sorted(glob(os.path.join(clear_dir, '*.bmp'))) + \
                           sorted(glob(os.path.join(blur_dir, '*.bmp'))) 

        self.clear_paths = sorted(glob(os.path.join(clear_dir, '*.bmp'))) 
        self.blur_paths = sorted(glob(os.path.join(blur_dir, '*.bmp'))) 
        # self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.bmp')))
        self.mask_paths1 = sorted(glob(os.path.join(mask_dir, '*.bmp')))
        self.mask_paths2 = ['image_white.bmp'] * len(self.clear_paths)
        self.mask_paths3 = ['image_black.bmp'] * len(self.blur_paths)
        self.mask_paths = self.mask_paths1 + self.mask_paths2 + self.mask_paths3
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        random_idx_c = random.randint(0, len(self.clear_paths) - 1)
        random_idx_b = random.randint(0, len(self.blur_paths) - 1)

        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        clear_path = self.clear_paths[random_idx_c]
        blur_path = self.blur_paths[random_idx_b]

        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        clear_img = cv2.imread(clear_path, cv2.COLOR_BGR2RGB)
        blur_img = cv2.imread(blur_path, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
            mask_img = self.transform(mask_img)
            clear_img = self.transform(clear_img)
            blur_img = self.transform(blur_img)

        return image, mask_img, clear_img, blur_img

    
def Create_nets():
    generator_clear = LaKDNet(inp_channels=4, out_channels=3, dim=16, num_blocks=[1, 2, 3, 4, 5], mix_kernel_size=[1, 3, 5, 7, 9],
                ffn_expansion_factor=2.3, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
    generator_blur = Generator_blur()
    discriminator_clear = MultiScalePatchDiscriminator()
    discriminator_blur = MultiScalePatchDiscriminator()

    if torch.cuda.is_available():
        generator_clear = generator_clear.cuda()
        generator_blur = generator_blur.cuda()
        discriminator_clear = discriminator_clear.cuda()
        discriminator_blur = discriminator_blur.cuda()     

    return generator_clear, generator_blur, discriminator_clear, discriminator_blur

def psnr_loss(output, target):

    output_clipped = torch.clamp(output, min=0.0, max=1.0)
    mse_loss = nn.MSELoss()(output_clipped, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
    return psnr

def save_image(images, output_dir, epoch, tag):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        if(tag == 'mask' or tag == 'mask2'):
            img = img.squeeze(0).cpu().numpy()  # Remove channel dimension if it exists
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img, mode='L')
        elif(tag[0] == 'o'):
            img = img.detach().cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy array
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img)
        else:
            img = img.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f'{i}_{tag}.png'))

def concat_images(file_path, batch_size):
    os.makedirs(file_path, exist_ok=True)
    for i in range(batch_size):
        images = [
            Image.open(os.path.join(file_path, str(i)+'_input.png')),
            Image.open(os.path.join(file_path, str(i)+'_output_img2clear.png')),
            Image.open(os.path.join(file_path, str(i)+'_output_clear2img.png')),
            
            Image.open(os.path.join(file_path, str(i)+'_mask.png')),
            Image.open(os.path.join(file_path, str(i)+'_output_img2blur.png')),
            Image.open(os.path.join(file_path, str(i)+'_output_blur2img.png')),
        ]

        width, height = images[0].size

        combined_width = width * 3
        combined_height = height * 2

        combined_image = Image.new('RGB', (combined_width, combined_height))

        for j, image in enumerate(images):
            x = (j % 3) * width
            y = (j // 3) * height
            combined_image.paste(image, (x, y))

        combined_image.save(os.path.join(file_path, 'img', f'{i}.png'))

def train_model(image_dir, mask_dir, clear_dir, blur_dir, epochs, batch_size, learning_rate, model_save_path, result_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])

    dataset = ImageDataset(
        image_dir, mask_dir, clear_dir, blur_dir, 
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator_clear, generator_blur, discriminator_clear, discriminator_blur = Create_nets()
    optimizer_G = optim.Adam(list(generator_clear.parameters()) + list(generator_blur.parameters()), lr=learning_rate, betas=(0.9, 0.999))
    optimizer_D_clear = optim.Adam(discriminator_clear.parameters(), lr=learning_rate)
    optimizer_D_blur = optim.Adam(discriminator_blur.parameters(), lr=learning_rate)

    generator_clear.train()
    generator_blur.train()
    discriminator_clear.train()
    discriminator_blur.train()

    for epoch in range(epochs):
        
        epoch_loss_gen = 0
        epoch_loss_D_clear = 0
        epoch_loss_D_blur = 0
        epoch_loss_cycle_b = 0
        epoch_loss_cycle_c = 0
        epoch_loss_G_clear2blur = 0
        epoch_loss_G_blur2clear = 0
        epoch_loss_unmasked = 0

        criterion_GAN = nn.BCEWithLogitsLoss()
        cycle_consistency_loss = nn.L1Loss()
        # adversarial_loss = nn.BCELoss()


        for img, mask_img, clear_img, blur_img in tqdm(dataloader):
            img, mask_img, clear_img, blur_img = img.to(device), mask_img.to(device), clear_img.to(device), blur_img.to(device)

            optimizer_G.zero_grad()
            optimizer_D_clear.zero_grad()
            optimizer_D_blur.zero_grad()

            valid = torch.ones((img.size(0), 1, 32, 32)).to(device) * 0.9
            fake = torch.zeros((img.size(0), 1, 32, 32)).to(device) * 0.1

            output_img2clear = generator_clear(img, 1-mask_img)
            output_clear2img = generator_blur(output_img2clear, 1-mask_img)

            output_img2blur = generator_blur(img, mask_img)
            output_blur2img = generator_clear(output_img2blur, mask_img)

            loss_cycle_c = cycle_consistency_loss(img, output_clear2img)
            loss_cycle_b = cycle_consistency_loss(img, output_blur2img)
            loss_unmasked = (cycle_consistency_loss(img*mask_img, output_img2clear*mask_img) \
                          + cycle_consistency_loss(output_img2clear*mask_img, output_clear2img*mask_img) \
                          + cycle_consistency_loss(img*(1-mask_img), output_img2blur*(1-mask_img)) \
                          + cycle_consistency_loss(output_img2blur*(1-mask_img), output_blur2img*(1-mask_img)))

            pred_clear = discriminator_clear(output_img2clear)
            pred_blur = discriminator_blur(output_img2blur)
            loss_blur2clear = criterion_GAN(pred_clear, valid) * 3
            loss_clear2blur = criterion_GAN(pred_blur, valid) * 3

            loss_G = loss_blur2clear + loss_clear2blur + loss_cycle_c + loss_cycle_b + loss_unmasked
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            
            # discriminator clear
            pred_real_clear = discriminator_clear(clear_img)
            loss_real_clear = criterion_GAN(pred_real_clear, valid)
            loss_fake_clear = criterion_GAN(discriminator_clear(output_img2clear.detach()), fake)

            loss_D_clear = 0.5 * (loss_real_clear + loss_fake_clear)

            loss_D_clear.backward(retain_graph=True)
            optimizer_D_clear.step()

            # discriminator blur
            pred_real_blur = discriminator_blur(blur_img)
            loss_real_blur = criterion_GAN(pred_real_blur, valid)
            loss_fake_blur = criterion_GAN(discriminator_blur(output_img2blur.detach()), fake)

            loss_D_blur = 0.5 * (loss_real_blur + loss_fake_blur)

            loss_D_blur.backward(retain_graph=True)
            optimizer_D_blur.step()


            epoch_loss_D_clear += loss_D_clear.item() / len(dataloader)
            epoch_loss_D_blur += loss_D_blur.item() / len(dataloader)
            epoch_loss_gen += loss_G.item() / len(dataloader)
            epoch_loss_cycle_c += loss_cycle_c.item() / len(dataloader)
            epoch_loss_cycle_b += loss_cycle_b.item() / len(dataloader)
            epoch_loss_G_clear2blur += loss_clear2blur.item() / len(dataloader)
            epoch_loss_G_blur2clear += loss_blur2clear.item() / len(dataloader)
            epoch_loss_unmasked += loss_unmasked.item() / len(dataloader)

            
        save_image(img, result_path, epoch, 'input')
        save_image(mask_img, result_path, epoch, 'mask')
        save_image(output_img2clear, result_path, epoch, 'output_img2clear')
        save_image(output_img2blur, result_path, epoch, 'output_img2blur')
        save_image(output_clear2img, result_path, epoch, 'output_clear2img')
        save_image(output_blur2img, result_path, epoch, 'output_blur2img')
        concat_images(result_path, batch_size)            

        os.makedirs(model_save_path, exist_ok=True)
        save_path_G_clear = os.path.join(model_save_path, f'generator_clear.pth')
        save_path_G_blur = os.path.join(model_save_path, f'generator_blur.pth')
        save_path_D_clear = os.path.join(model_save_path, f'discriminator_clear.pth')
        save_path_D_blur = os.path.join(model_save_path, f'discriminator_blur.pth')
        torch.save(generator_clear.state_dict(), save_path_G_clear)
        torch.save(generator_blur.state_dict(), save_path_G_blur)
        torch.save(discriminator_clear.state_dict(), save_path_D_clear)
        torch.save(discriminator_blur.state_dict(), save_path_D_blur)
        generator_blur.eval()
        generator_clear.eval()
        discriminator_clear.eval()
        discriminator_blur.eval()
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Total: {epoch_loss_gen:.4f} | mask: {epoch_loss_unmasked:.4f} | "
              f"b2c: {epoch_loss_G_blur2clear:.4f} | c2b: {epoch_loss_G_clear2blur:.4f} | ",
              f"cycle c: {epoch_loss_cycle_c:.4f} | cycle b: {epoch_loss_cycle_b:.4f} | ",
              f"Dc: {epoch_loss_D_clear:.4f} | Db: {epoch_loss_D_blur:.4f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a simple image generator.')
    parser.add_argument('--image_dir', type=str, default='/home/oscar/Desktop/dataset/SG_dataset/dataset/train_data/1204source')
    parser.add_argument('--mask_dir', type=str, default='/home/oscar/Desktop/dataset/SG_dataset/dataset/train_data/1204gt')
    parser.add_argument('--clear_path', type=str, default='/home/oscar/Desktop/dataset/SG_dataset/dataset/train_data/FCFB/FCC/FC/')
    parser.add_argument('--blur_path', type=str, default='/home/oscar/Desktop/dataset/SG_dataset/dataset/train_data/FCFB/FBB/FB/')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')
    parser.add_argument('--model_save_path', type=str, default='./0730_result/1/saved_models', help='Path to save the trained model.')
    parser.add_argument('--result_path', type=str, default='./0730_result/3')

    args = parser.parse_args()

    train_model(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        clear_dir=args.clear_path,
        blur_dir=args.blur_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=args.model_save_path,
        result_path=args.result_path
    )