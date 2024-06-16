import argparse
import os
import torch
from tqdm import tqdm
from my_utils import my_to_image_test, to_image
from torch.autograd import Variable
from my_datasets import Get_dataloader_test
from models.my_SG import GeneratorUNet
device = torch.device("cuda:0")

def test(stict,mask_save_path,image_path):

    if not torch.cuda.is_available():
        generator2 = GeneratorUNet()
    else:
        generator2 = GeneratorUNet().cuda()
    generator2.load_state_dict(torch.load(stict))
    generator2.eval()

    dataloder = Get_dataloader_test(image_path, 1)
    for i,(img,index) in tqdm(enumerate(dataloder)):
        # if not torch.cuda.is_available():
        #     img=Variable(img)
        # else:
        #     img = Variable(img).cuda()
        mask = generator2(img)

        os.makedirs(mask_save_path, exist_ok=True)
        # my_to_image_test(mask, i=int(index.data.numpy()),tag='', path=mask_save_path)
        to_image(mask, i=i, tag='output', path=mask_save_path,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--stict', default='./my_log/SG-0312_DPDD/saved_models/generator_332000.pth',type=str)
    parser.add_argument('--image_path', default='/home/oscar/Desktop/dataset/DPDD/dd_dp_dataset_png/train_c/source', type=str)
    parser.add_argument('--mask_save_path', default='./results_0430/', type=str)
    args=parser.parse_args()
    #test
    test(args.stict,args.mask_save_path,args.image_path)