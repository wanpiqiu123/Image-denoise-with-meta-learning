from PIL import Image
import sys
import os
import random
#切图
def cut_image(image):
    width, height = image.size
    # print(image.size)
    item_width = 256
    box_list = []    
    # (left, upper, right, lower)
    for i in range(1):
        r_width = random.uniform(0.1,0.9)*(width-item_width)
        r_height = random.uniform(0.1,0.9)*(height-item_width)
        box = (r_width,r_height,r_width+item_width,r_height+item_width)
        box_list.append(box)
    # for i in range(0,3):#两重循环，生成9张图片基于原图的位置
    #     for j in range(0,3):           
    #         #print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
    #         box = (j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
    #         box_list.append(box)
    # image_list = [image.crop(box).resize((40,40)) for box in box_list]    
    image_list = [image.crop(box) for box in box_list]    

    return image_list

#保存
def save_images(image_list,name):
    index = 1
    dir_name = './Kodak24_n/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for image in image_list:
        # image.show()
        image.save(dir_name+name+'_'+str(index) + '.png', 'PNG')
        index += 1

if __name__ == '__main__':
    # file_path = "./augment/pic/im1.jpg"
    # image = Image.open(file_path)   
    # #image.show()
    # image_list = cut_image(image)
    # save_images(image_list,1)
    file_dir = "./Kodak24"
    for root, dirs, files in os.walk(file_dir):
        for f in files:
            name=f.split('.')[0]
            # n=list(filter(str.isdigit,f))
            # n="".join(n)
            # print(int(n))
            # print(os.path.join(file_dir,f))
            image = Image.open(os.path.join(file_dir,f))
            # print(name)
            image_list = cut_image(image)
            save_images(image_list,name)