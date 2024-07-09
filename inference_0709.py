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
from LaKDNet import LaKDNet
import argparse

class ImageDataset_test(Dataset):
    def __init__(self, image_dir, mask_dir, mask2_dir, transform):
        
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.png')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.mask2_paths = sorted(glob(os.path.join(mask2_dir, '*.png')))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        mask2_path = self.mask2_paths[idx]

        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask2_img = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            mask_img = self.transform(mask_img)
            mask2_img = self.transform(mask2_img)
        return image, mask_img, mask2_img
    
def dataloader_test(image_dir, mask_dir, mask2_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])

    dataset = ImageDataset_test(
        image_dir, mask_dir, mask2_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

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

def concat_images(file_path, idx):
    # for i in range(1):
    images = [
        Image.open(os.path.join(file_path, str(0)+'_input.png')),    
        Image.open(os.path.join(file_path, str(0)+'_output.png')),
    
        Image.open(os.path.join(file_path, str(0)+'_mask.png')),
        Image.open(os.path.join(file_path, str(0)+'_mask2.png')),
    ]

    # 假設所有圖片大小相同，取得圖片大小
    width, height = images[0].size

    # 設定合併後的圖片大小，這裡是 4x2 的佈局
    combined_width = width * 2
    combined_height = height * 2

    # 創建一個新的空白圖片
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # 將每張圖片粘貼到新的空白圖片上
    for j, image in enumerate(images):
        x = (j % 2) * width
        y = (j // 2) * height
        combined_image.paste(image, (x, y))

    # 保存合併後的圖片
    combined_image.save(os.path.join(file_path, 'img', f'{idx}.png'))

def test(genC, genB, result_path, image_path, mask_path, mask2_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        generator_clear = LaKDNet(inp_channels=4, out_channels=3, dim=16, num_blocks=[1, 2, 2, 3], mix_kernel_size=[1, 3, 5, 7],
                ffn_expansion_factor=2.3, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
        generator_blur = Generator()
    else:
        generator_clear = LaKDNet(inp_channels=4, out_channels=3, dim=16, num_blocks=[1, 2, 2, 3], mix_kernel_size=[1, 3, 5, 7],
                ffn_expansion_factor=2.3, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False).cuda()
        generator_blur = Generator().cuda()

    generator_clear.load_state_dict(torch.load(genC))
    generator_clear.eval()
    generator_blur.load_state_dict(torch.load(genB))
    generator_blur.eval()

    dataloader = dataloader_test(image_path, mask_path, mask2_path, 1)

    for i, (img, mask_img, mask2_img) in tqdm(enumerate(dataloader)):
        img, mask_img, mask2_img = img.to(device), mask_img.to(device), mask2_img.to(device)

        output = generator_blur(img, mask_img)
        output = generator_clear(output, mask2_img)

        # output2 = generator_clear(mask2_path)
        # output2 = generator_blur(mask_path)

        os.makedirs(result_path, exist_ok=True)
        save_image(img, result_path, 0, 'input')
        save_image(mask_img, result_path, 0, 'mask')
        save_image(mask2_img, result_path, 0, 'mask2')
        save_image(output, result_path, 0, 'output')
        concat_images(result_path, i) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--genC', default='./0709_result/3/save/saved_models_cycle/generator_clear.pth',type=str)
    parser.add_argument('--genB', default='./0709_result/3/save/saved_models_cycle/generator_blur.pth',type=str)
    parser.add_argument('--image_path', default='/home/oscar/Desktop/dataset/DPDD/dd_dp_dataset_png/train_c/source', type=str)
    parser.add_argument('--mask_path', default='/home/oscar/Desktop/SG/source_map', type=str)
    parser.add_argument('--mask2_path', default='/home/oscar/Desktop/0301_dpdd_seg/dpdd_seg_mask_copy', type=str)
    parser.add_argument('--mask_save_path', default='./0709_result/test', type=str)
    args=parser.parse_args()
    #test
    test(args.genC, args.genB, args.mask_save_path,args.image_path, args.mask_path, args.mask2_path)

