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
from SRGAN_model import Generator as Generator_clear
from LaKDNet import LaKDNet
  
class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, clear_dir, blur_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.bmp')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.bmp')))
        self.clear_paths = sorted(glob(os.path.join(clear_dir, '*.bmp'))) 
        self.blur_paths = sorted(glob(os.path.join(blur_dir, '*.bmp'))) 
        
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



class ClearImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = glob(os.path.join(dataset, '*.bmp')) 
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
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
        self.In = nn.InstanceNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.In(input)
        if self.activation:
            input = self.activation(input)

        return input
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            BaseConv(4, 64, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            BaseConv(64, 128, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            BaseConv(128, 256, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            # BaseConv(256, 256, 1, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
        )

        self.decoder = nn.Sequential(
            # BaseConv(512, 256, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            # BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            BaseConv(256, 128, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            BaseConv(128, 64, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            BaseConv(64, 32, 3, 1, activation=nn.ReLU(inplace=False), use_bn=True),
            BaseConv(32, 3, 3, 1, activation=None, use_bn=False),
        )

    def forward(self, real_image, mask):
    # def forward(self, masked_image):
        input_combined = torch.cat([real_image, mask], dim=1)
        encoder = self.encoder(input_combined)
        # encoder = self.encoder(masked_image)
        
        decoder = self.decoder(encoder)

        return decoder

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            BaseConv(3, 64, 3, 2, activation=nn.LeakyReLU(0.2,inplace=False), use_bn=False),  # Input: 3x256x256, Output: 64x128x128
            BaseConv(64, 128, 3, 2, activation=nn.LeakyReLU(0.2,inplace=False), use_bn=True),  # Output: 128x64x64
            BaseConv(128, 256, 3, 4, activation=nn.LeakyReLU(0.2,inplace=False), use_bn=True),  # Output: 256x32x32
            BaseConv(256, 256, 3, 4, activation=nn.LeakyReLU(0.2,inplace=False), use_bn=True),  # Output: 512x16x16
            # BaseConv(512, 512, 3, 2, activation=nn.LeakyReLU(0.2,inplace=False), use_bn=True),  # Output: 512x8x8
            # BaseConv(512, 512, 3, 2, activation=nn.LeakyReLU(0.2,inplace=False), use_bn=True),  # Output: 512x4x4
            # BaseConv(512, 512, 3, 1, activation=nn.LeakyReLU(0.2,inplace=False), use_bn=True),  # Output: 512x4x4
            nn.Conv2d(256, 1, 4, 1, 0),  # Output: 1x1x1 (Real/Fake)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x).view(-1, 1)

class Discriminator_SG(nn.Module):
    def __init__(self,opt=None):
        super(Discriminator_SG,self).__init__()
        self.ngpu = 1
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),


            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data,torch.cuda.FloatTensor) and self.ngpu>1:
            output = nn.parallel.data_parallel(self.net,x,range(self.ngpu))
        else:
            output = self.net(x)
        return output
    
def Create_nets():
    # generator_clear = Generator_clear()
    generator_clear = LaKDNet(inp_channels=4, out_channels=3, dim=16, num_blocks=[1, 2, 2, 3], mix_kernel_size=[1, 3, 5, 7],
                ffn_expansion_factor=2.3, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
    generator_blur = Generator()
    discriminator_clear = Discriminator_SG()
    discriminator_blur = Discriminator_SG()

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
    for i in range(batch_size):
        images = [
            Image.open(os.path.join(file_path, str(i)+'_input.png')),
            
            Image.open(os.path.join(file_path, str(i)+'_output_img2clear.png')),
            Image.open(os.path.join(file_path, str(i)+'_output_clear2img.png')),
            
            Image.open(os.path.join(file_path, str(i)+'_mask.png')),

            # Image.open(os.path.join(file_path, str(i)+'_output.png')),
            Image.open(os.path.join(file_path, str(i)+'_output_img2blur.png')),
            Image.open(os.path.join(file_path, str(i)+'_output_blur2img.png')),
            
            # Image.open(os.path.join(file_path, str(i)+'_gt.png')),
            # Image.open('/home/oscar/Desktop/Focus/gray_image.png'),
            
        ]

        # 假設所有圖片大小相同，取得圖片大小
        width, height = images[0].size

        # 設定合併後的圖片大小，這裡是 4x2 的佈局
        combined_width = width * 3
        combined_height = height * 2

        # 創建一個新的空白圖片
        combined_image = Image.new('RGB', (combined_width, combined_height))

        # 將每張圖片粘貼到新的空白圖片上
        for j, image in enumerate(images):
            x = (j % 3) * width
            y = (j // 3) * height
            combined_image.paste(image, (x, y))

        # 保存合併後的圖片
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


    # generator_clear = LaKDNet(inp_channels=4, out_channels=3, dim=16, num_blocks=[1, 2, 2, 3], mix_kernel_size=[1, 3, 5, 7],
    #             ffn_expansion_factor=2.3, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
    # generator_clear = generator_clear.to('cuda' if torch.cuda.is_available() else 'cpu')

    generator_clear, generator_blur, discriminator_clear, discriminator_blur = Create_nets()
    optimizer_G = optim.Adam(list(generator_clear.parameters()) + list(generator_blur.parameters()), lr=learning_rate, betas=(0.9, 0.999))
    optimizer_D_clear = optim.Adam(discriminator_clear.parameters(), lr=learning_rate)
    optimizer_D_blur = optim.Adam(discriminator_blur.parameters(), lr=learning_rate)

    # G_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)
    # Dc_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_D_clear, step_size=5, gamma=0.5)
    # Db_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_D_blur, step_size=5, gamma=0.5)

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

        criterion_GAN = nn.MSELoss()
        cycle_consistency_loss = nn.L1Loss()
        adversarial_loss = nn.BCEWithLogitsLoss()


        for img, mask_img, clear_img, blur_img in tqdm(dataloader):
            img, mask_img, clear_img, blur_img = img.to(device), mask_img.to(device), clear_img.to(device), blur_img.to(device)

            optimizer_G.zero_grad()
            optimizer_D_clear.zero_grad()
            optimizer_D_blur.zero_grad()

            valid = torch.ones((img.size(0), 1)).to(device)
            fake = torch.zeros((img.size(0), 1)).to(device)

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

            # G_scheduler.step()
            # Dc_scheduler.step()
            # Db_scheduler.step()

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
    parser.add_argument('--model_save_path', type=str, default='./saved_models_cycle', help='Path to save the trained model.')
    parser.add_argument('--result_path', type=str, default='./0716_result/1')

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
