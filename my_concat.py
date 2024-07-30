import os
from PIL import Image
# import numpy as np

def concat_images(file_path):
    gray_value = 128
    # image = np.full((256, 256), 128, dtype=np.uint8)
    image = Image.new('L', (256, 256), color=gray_value)
    image.save('gray_image.png')
    for i in range(350):
        images = [
            Image.open(os.path.join(file_path, "origin", "input", str(i)+'_input.png')),
            Image.open(os.path.join(file_path, "mask2", str(i)+'_mask2.png')),
            Image.open('gray_image.png'),
            Image.open(os.path.join(file_path, "mask", str(i)+'_mask.png')),

            Image.open(os.path.join(file_path, "output2", str(i)+'_output2.png')),
            Image.open(os.path.join(file_path, "SG", "output2", str(i)+'.bmp')),
            Image.open(os.path.join(file_path, "output", str(i)+'_output.png')),
            Image.open(os.path.join(file_path, "SG", "output", str(i)+'.bmp')),
        ]

        # 假設所有圖片大小相同，取得圖片大小
        width, height = images[0].size

        # 設定合併後的圖片大小，這裡是 4x2 的佈局
        combined_width = width * 4
        combined_height = height * 2

        # 創建一個新的空白圖片
        combined_image = Image.new('RGB', (combined_width, combined_height))

        # 將每張圖片粘貼到新的空白圖片上
        for j, image in enumerate(images):
            x = (j % 4) * width
            y = (j // 4) * height
            combined_image.paste(image, (x, y))

        # 保存合併後的圖片
        combined_image.save(os.path.join('/home/oscar/Desktop/SG/0611_result/2/SG/concat_train_segmask', f'{i}.png'))

concat_images("/home/oscar/Desktop/SG/0611_result/2/SG")