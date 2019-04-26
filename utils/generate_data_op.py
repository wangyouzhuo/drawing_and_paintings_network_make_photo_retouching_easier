from PIL import Image
import os
import numpy as np
from utils.load_image import *



def generate_data(loadpath,savepath):
    all_count = len(os.listdir(loadpath))
    count = 0
    for pic_name in os.listdir(loadpath):
        count = count + 1
        print("当前读取：%s || 进度:%s%%" % (count,round((count/all_count)*100.0,4)))
        picture = Image.open(loadpath + pic_name)
        shape = picture.size
        img_np = np.array(picture)

        image = load_image2numpy(loadpath + pic_name)
        whe = check_histogram(image)

        error_rg = np.sum(image[300:400, 300:400, 0] - image[300:400, 300:400, 1])  # 排除黑白照片
        error_gb = np.sum(image[300:400, 300:400, 1] - image[300:400, 300:400, 2])  # 排除黑白照片

        target_size = (512, 512)
        if whe and error_rg > 10 and error_gb>10 :
            height = shape[1]  # 图片高度
            width = shape[0]  # 图片宽度
            if height > width:
                x = 0
                w = width
                h = width
                y_list = [0, 0.4 * w, height - width]
                i = 0
                for y in y_list:
                    i = i + 1
                    print("        裁剪第：%s 次" % i)
                    region = picture.crop((x, y, x + w, y + h))
                    region = region.resize(target_size)
                    region.save(savepath +'No.' +str(count)+"-已裁剪" + str(i) + "次" +'.jpg')
            else:
                y = 0
                w = height
                h = height
                x_list = [0, 0.25 * w, width - height]
                i = 0
                for x in x_list:
                    i = i + 1
                    print("        裁剪第：%s 次" % i)
                    region = picture.crop((x, y, x + w, y + h))
                    region = region.resize(target_size)
                    region.save(savepath +'No.' +str(count)+"-已裁剪" + str(i) + "次" +'.jpg')



