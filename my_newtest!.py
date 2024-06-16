import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
from glob import glob
import os
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, mask2_dir, gt_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.png')))  # Adjust extension if needed
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))    # Adjust extension if needed
        self.gt_paths = sorted(glob(os.path.join(gt_dir, '*.png')))        # Adjust extension if needed
        self.mask2_paths = sorted(glob(os.path.join(mask2_dir, '*.png')))
        assert len(self.image_paths) == len(self.mask_paths) == len(self.gt_paths) == len(self.mask2_paths), \
            "The number of images, masks, and ground truth images must be the same"
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        gt_path = self.gt_paths[idx]
        mask2_path = self.mask2_paths[idx]

        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.COLOR_BGR2RGB)
        mask2_img = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)
            mask_img = self.transform(mask_img)
            gt_img = self.transform(gt_img)
            mask2_img = self.transform(mask2_img)

        return image, mask_img, mask2_img, gt_img
    
def save_image(images, output_dir, epoch, tag):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        if(tag == 'mask' or tag == 'mask2'):
            img = img.squeeze(0).cpu().numpy()  # Remove channel dimension if it exists
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img, mode='L')
        elif(tag == 'output'or tag == 'clear' or tag == 'blur' or tag == 'output2'):
            img = img.detach().cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy array
            # img = img.permute(1, 2, 0)
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img)
        else:
            img = img.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
            img = (img * 255).astype(np.uint8)  # Convert to uint8 type
            img = Image.fromarray(img)
        img.save(os.path.join(output_dir, f'{epoch}_{tag}.png'))

def concat_images(file_path):
    for i in range(350):
        images = [
            Image.open(os.path.join(file_path, str(i)+'_input.png')),
            Image.open(os.path.join(file_path, str(i)+'_output.png')),
            Image.open(os.path.join(file_path, str(i)+'_output2.png')),

            Image.open(os.path.join(file_path, str(i)+'_mask.png')),
            Image.open(os.path.join(file_path, str(i)+'_mask2.png')),
            Image.open(os.path.join(file_path, str(i)+'_gt.png')),
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
        combined_image.save(os.path.join('/home/oscar/Desktop/SG/0611_result/2/concat_rev', f'{i}.png'))


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
        )

        self.decoder = nn.Sequential(
            BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True),
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

# Define the transformation for preprocessing input images




# Define a function for performing inference
def inference(input_image_path, mask_path, mask2_path, gt_path, generator_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])

    dataset = ImageDataset(
        input_image_path, mask_path, mask2_path, gt_path,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    # Load the generator model
    generator = Generator()
    generator.load_state_dict(torch.load(generator_model_path))
    generator.eval()

    if torch.cuda.is_available():
        generator = generator.cuda()
    # print(generator)

    i = 0
    for img, mask_img, mask2_img, gt_img in tqdm(dataloader):
        img, mask_img, mask2_img, gt_img  = img.to(device), mask_img.to(device), mask2_img.to(device), gt_img.to(device),

        with torch.no_grad():
            output = generator(img, mask_img)
            output2 = generator(output, mask2_img)


        # save_image(output, '/home/oscar/Desktop/SG/0611_result/2/test', i, 'output')
        # save_image(img, '/home/oscar/Desktop/SG/0611_result/2/test', i, 'input')
        # save_image(mask_img, '/home/oscar/Desktop/SG/0611_result/2/test', i, 'mask')
        # save_image(mask2_img, '/home/oscar/Desktop/SG/0611_result/2/test', i, 'mask2')
        # save_image(output2, '/home/oscar/Desktop/SG/0611_result/2/test', i, 'output2')
        # save_image(gt_img, '/home/oscar/Desktop/SG/0611_result/2/test', i, 'gt')
        save_image(output, '/home/oscar/Desktop/SG/0611_result/2/SG/output', i, 'output')
        save_image(img, '/home/oscar/Desktop/SG/0611_result/2/SG/input', i, 'input')
        save_image(mask_img, '/home/oscar/Desktop/SG/0611_result/2/SG/mask', i, 'mask')
        save_image(mask2_img, '/home/oscar/Desktop/SG/0611_result/2/SG/mask2', i, 'mask2')
        save_image(output2, '/home/oscar/Desktop/SG/0611_result/2/SG/output2', i, 'output2')
        save_image(gt_img, '/home/oscar/Desktop/SG/0611_result/2/SG/gt', i, 'gt')
        i += 1

    concat_images('./0611_result/2/test')

# Example usage
input_image_path = '/home/oscar/Desktop/dataset/DPDD/dd_dp_dataset_png/train_c/source'  # Path to the input RGB image
mask_path = '/home/oscar/Desktop/0301_dpdd_seg/dpdd_seg_mask_copy'  # Path to the mask image
mask2_path = '/home/oscar/Desktop/SG/source_map'  # Path to the mask image
gt_path = '/home/oscar/Desktop/dataset/DPDD/dd_dp_dataset_png/train_c/target'  # Path to the mask image
generator_model_path = './saved_models/generator_500.pth'  # Path to the trained generator model

output_image = inference(input_image_path, mask_path, mask2_path, gt_path, generator_model_path)

# Display or save the output image as needed
# cv2.imshow('Output Image', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
