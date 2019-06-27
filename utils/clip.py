from PIL import Image
import os

# 存放源图片的路径：
source_path = "/home/wyz/桌面/富士film-c200/"

# 存放裁剪后图片的路径
result_path = "/home/wyz/桌面/裁剪-富士"

# 裁剪后的边长
target_size = 224

for pic_name in os.listdir(source_path):
    print("当前裁剪：%s"%pic_name)
    picture = Image.open(source_path+"/"+pic_name)
    shape = picture.size
    height = shape[1]  # 图片高度
    width = shape[0]  # 图片宽度
    if height>width:
        x= 0
        w = width
        h = width
        y_list = [0,height-width,0.6*(height-width)]
        i = 0
        for y in y_list:
            i = i+1
            print("        裁剪第：%s 次"%i)
            region = picture.crop((x, y, x + w, y + h))
            region = region.resize((target_size, target_size))
            region.save(result_path+"/已裁剪"+str(i)+"次"+pic_name)
    else:
        y = 0
        w = height
        h = height
        x_list = [0, width-height,0.6*(width-height)]
        i = 0
        for x in x_list:
            i = i + 1
            print("        裁剪第：%s 次"%i)
            region = picture.crop((x, y, x + w, y + h))
            region = region.resize((target_size, target_size))
            region.save(result_path + "/已裁剪" +str(i) + "次" + pic_name)