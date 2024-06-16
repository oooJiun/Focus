from my_loss import *
from models.my_models import Create_nets
from my_datasets import *
from my_options import TrainOptions
from optimizer import *
from test import test
from eval import eval1
from my_utils import *
import torch.nn as nn
import torch
import PIL.Image as pil
from my_ssim import *
from DepthAnything.my_run import *
import matplotlib.pyplot as plt
import numpy as np
def show_image(image_tensor):
    image_np = image_tensor.cpu().detach().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
#load the args
args = TrainOptions().parse()
# from models.my_SG import my_Generator

########
checkpoint = torch.load('my_log/SG-0312_DPDD/saved_models/generator_83000.pth')
generator, discriminator,discriminator2 = Create_nets(args)

optimizer = YourOptimizerClass(model.parameters())

criterion = YourCriterionClass()

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
criterion.load_state_dict(checkpoint['criterion_state_dict'])

########

# generator = my_Generator()
# generator.cuda()
# Loss functions
criterion_GAN, criterion_pixelwise = Get_loss_func(args)
# new_image_loss_fn = MaskLoss(args.min_mask_coverage, args.mask_alpha, args.binarization_alpha)
# new_image_loss_fn = PerceptualLoss()

# Optimizers
optimizer_G, optimizer_D,optimizer_D2 = Get_optimizers(args, generator, discriminator,discriminator2)
log={'bestmae_it':0,'best_mae':10,'fm':0,'bestfm_it':0,'best_fm':0,'mae':0}
# Configure dataloaders
# dataloader = Get_dataloader_pair(args.path_gt, args.path_mask, args.batch_size) #sourse

dis_C_loder1 = Get_dataloader(args.path_clear, args.batch_size)     #all clear
dis_C_loder2 = Get_dataloader(args.path_clear, args.batch_size)     #all clear

dis_B_loder1 = Get_dataloader(args.path_blur, args.batch_size)     #all blur
dis_B_loder2 = Get_dataloader(args.path_blur, args.batch_size)     #all blur

# real_loader, mask_loader = Get_dataloader(args.path_gt, args.batch_size)  
img_loader = Get_dataloader(args.path_gt, args.batch_size)  

# mask_loader = Get_dataloader(args.path_mask, args.batch_size)      #mask

# real, masks = iter(dataloader)
dis_C1 = iter(dis_C_loder1)
dis_C2 = iter(dis_C_loder2)

dis_B1 = iter(dis_B_loder1)
dis_B2 = iter(dis_B_loder2)
# real = iter(real_loader)

# mask = iter(mask_loader)

img = iter(img_loader)

j=0
# 开始训练
pbar = range(args.epoch_start,3_000_000)
# pbar = range(0,1)
for i in pbar:
    try:
        real_image, mask_image, gt_image = next(img)

        # real_image = next(real)
        real_image = real_image.to(device)

        dis_C_image1 = next(dis_C1)
        dis_C_image1 = dis_C_image1.to(device)
        dis_C_image2 = next(dis_C2)
        dis_C_image2 = dis_C_image2.to(device)

        dis_B_image1 = next(dis_B1)
        dis_B_image1 = dis_B_image1.to(device)
        dis_B_image2 = next(dis_B2)
        dis_B_image2 = dis_B_image2.to(device)

        # mask_image= next(mask)
        mask_image = mask_image.to(device)

        gt_image = gt_image.to(device)

    except (OSError, StopIteration):
        img = iter(img_loader)
        # real = iter(real_loader)
        dis_C1 = iter(dis_C_loder1)
        dis_C2 = iter(dis_C_loder2)

        dis_B1 = iter(dis_B_loder1)
        dis_B2 = iter(dis_B_loder2)

        # mask_image = iter(mask)
        # mask = iter(mask_loader)

        real_image, mask_image, gt_image = next(img)
        # real_image = next(real)
        real_image = real_image.to(device)

        dis_C_image1 = next(dis_C1)
        dis_C_image1 = dis_C_image1.to(device)
        dis_C_image2 = next(dis_C2)
        dis_C_image2 = dis_C_image2.to(device)

        dis_B_image1 = next(dis_B1)
        dis_B_image1 = dis_B_image1.to(device)
        dis_B_image2 = next(dis_B2)
        dis_B_image2 = dis_B_image2.to(device)

        # mask_image= next(mask)
        mask_image = mask_image.to(device)
        gt_image = gt_image.to(device)


    # ------------------
    #  Train Generators
    # ------------------

    # Adversarial ground truths
    patch=(1,1,1)
    valid = Variable(torch.FloatTensor(np.ones((real_image.size(0),*patch))).cuda(), requires_grad=False)
    fake = Variable(torch.FloatTensor(np.zeros((real_image.size(0),*patch))).cuda(), requires_grad=False)

    optimizer_G.zero_grad()
    requires_grad(generator, True)
    requires_grad(discriminator, False)
    requires_grad(discriminator2, False)
    
    new_image = generator(real_image, mask_image)
    to_image(real_image, i=0, tag='input', path='my_temp/')
    to_image(new_image, i=1, tag='output', path='my_temp/')
    # depth_estimation1 = process_image('my_temp/0_input.png', 0)
    # depth_estimation2 = process_image('my_temp/1_output.png', 1)
    # depth_estimation1 = torch.tensor(depth_estimation1, dtype=torch.float32)
    # depth_estimation2 = torch.tensor(depth_estimation2, dtype=torch.float32)
    # criterion = nn.MSELoss()
    # depth_loss = criterion(depth_estimation1, depth_estimation2)*0.0001
    # depth_loss = min(abs(depth_loss), 0.5)-0.5
    depth_loss = 0

    syn_image_clear = mask_image * new_image + (1 - mask_image) * dis_C_image1
    syn_image_blur = mask_image * dis_B_image1 + (1 - mask_image) * new_image

    pred_fake = discriminator(syn_image_clear)
    loss_GAN1 = criterion_GAN(pred_fake, valid)
    pred_fake2 = discriminator2(syn_image_blur)
    loss_GAN2 = criterion_GAN(pred_fake2, valid)
    # loss_new_image = new_image_loss_fn(new_image, real_image)
    # ssim_loss = (1-ssim(new_image, real_image))*2
    color_loss = color_preservation_loss(gt_image, new_image)
    
    loss_c,loss_b,c,b = feather_loss(syn_image_clear, dis_C_image2, syn_image_blur, dis_B_image2)

    # Total loss
    loss_G = 0.01*(loss_GAN1+loss_GAN2)+depth_loss+(loss_c+loss_b)+color_loss
    # loss_G=loss_GAN2+loss_mask+loss_GAN1
    loss_G.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # # ---------------------
    optimizer_D.zero_grad()
    requires_grad(generator, False)
    requires_grad(discriminator, True)
    requires_grad(discriminator2, False)
    # Real loss
    pred_real = discriminator(dis_C_image2)
    loss_real = criterion_GAN(pred_real, valid)

    # Fake loss
    pred_fake = discriminator(syn_image_clear.detach())
    loss_fake = criterion_GAN(pred_fake, fake)
    # print(pred_fake)
    # print(loss_fake)

    # Total loss
    loss_D = 0.5 * (loss_real + loss_fake)*10
    loss_D.backward()
    optimizer_D.step()

    optimizer_D2.zero_grad()
    requires_grad(generator, False)
    requires_grad(discriminator, False)
    requires_grad(discriminator2, True)
    # Real loss
    pred_real2 = discriminator2(dis_B_image2)
    loss_real2 = criterion_GAN(pred_real2, valid)

    # Fake loss
    pred_fake2 = discriminator2(syn_image_blur.detach())
    loss_fake2 = criterion_GAN(pred_fake2, fake)

    # Total loss
    loss_D2 = 0.5 * (loss_real2 + loss_fake2)
    loss_D2.backward()
    optimizer_D2.step()

    if i%1000==0:
        # print(
        #     "\r[Batch%d]-[Total_loss:%f]-[Dloss:%f,Dloss2:%f]-[depth_loss:NaN, color_loss:%f, loss_GAN1:%f, loss_GAN2:%f]" %
        #     (i,loss_G.data.cpu(),loss_D.data.cpu(),loss_D2.data.cpu(),
        #      color_loss.data.cpu(), loss_GAN1.data.cpu(),loss_GAN2.data.cpu()))
        with open("output_log.txt", "a") as file:
            output = "\r[Batch%d]-[Total_loss:%f]-[Dloss:%f,Dloss2:%f]-[color_loss:%f, loss_GAN1:%f, loss_GAN2:%f]" %\
            (i,loss_G.data.cpu(),loss_D.data.cpu(),loss_D2.data.cpu(),
             color_loss.data.cpu(), loss_GAN1.data.cpu(),loss_GAN2.data.cpu())
            
            print(output)
            file.write(output + "\n")

        image_path = 'my_log/%s-%s/%s' % (args.exp_name, args.dataset_name, args.img_result_dir)
        os.makedirs(image_path, exist_ok=True)
        to_image(real_image, i=i, tag='input', path=image_path)
        to_image(mask_image, i=i, tag='mask', path=image_path)
        to_image(syn_image_clear, i=i, tag='syn_image', path=image_path)
        to_image(syn_image_blur, i=i, tag='syn_blur', path=image_path)
        to_image(new_image, i=i, tag='output', path=image_path)
        # to_image_mask(mask, i=i, tag='mask', path=image_path)

    if args.checkpoint_interval != -1 and i % 1000== 0:
    # Save model checkpoints
        # torch.save(generator.state_dict(), 'my_log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,i))
        # torch.save(discriminator.state_dict(), 'my_log/%s-%s/%s/discriminator1_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))
        # torch.save(discriminator2.state_dict(), 'my_log/%s-%s/%s/discriminator2_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))
        torch.save({
            'epoch': i,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_G.state_dict(),
            'criterion_state_dict': criterion_GAN.state_dict(),
            'lr': optimizer_G.param_groups[0]['lr'],  # Save learning rate
        }, 'my_log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,i))

        torch.save({
            'epoch': i,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_D.state_dict(),
            'lr': optimizer_D.param_groups[0]['lr'],  # Save learning rate
        }, 'my_log/%s-%s/%s/discriminator1_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))

        torch.save({
            'epoch': i,
            'model_state_dict': discriminator2.state_dict(),
            'optimizer_state_dict': optimizer_D2.state_dict(),
            'lr': optimizer_D2.param_groups[0]['lr'],  # Save learning rate
        }, 'my_log/%s-%s/%s/discriminator2_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir, i))
        # pthpath='my_log/%s-%s/%s/generator_%d.pth' % (args.exp_name, args.dataset_name,args.model_result_dir,i)
        # mask_save_path = 'my_log/%s-%s/test/test100-%s' % (args.exp_name, args.dataset_name, i)
        # # image_path= './dataset/test/xu100-source'
        # image_path= '../dataset/SG_dataset/dataset/test_data/CUHK/xu100-source'
        # test(pthpath,mask_save_path,image_path)

        # # gt_path = './dataset/test/xu100-gt'
        # gt_path = '../dataset/SG_dataset/dataset/test_data/CUHK/xu100-gt'
        # mae1,fmeasure1,_,_=eval1(mask_save_path,gt_path,1.5)

        # if mae1<log['best_mae'] :
        #     log['bestmae_it']=i
        #     log['best_mae']=mae1
        #     log['fm']=fmeasure1
        # if fmeasure1>log['best_fm']:
        #     log['bestfm_it']=i
        #     log['best_fm']=fmeasure1
        #     log['mae']=mae1
        # print('====================================================================================================================')
        # print('batch:',i, "mae:", mae1, "fmeasure:", fmeasure1)
        # print('bestmae_it',log['bestmae_it'],'best_mae',log['best_mae'],'fm:',log['fm'])
        # print('bestfm_it',log['bestfm_it'],'mae:',log['mae'],'best_fm',log['best_fm'])
        # print('=====================================================================================================================')