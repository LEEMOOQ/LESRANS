from PIL import Image
import matplotlib.pyplot as plt


def show_cut(path, left, upper, right, lower):
    """
        原图与所截区域相比较
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    """

    img = Image.open(path)

    print("This image's size: {}".format(img.size))  # (W, H)

    plt.figure("Image Contrast")

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(img)
    plt.axis('off')

    box = (left, upper, right, lower)
    roi = img.crop(box)

    plt.subplot(1, 2, 2)
    plt.title('roi')
    plt.imshow(roi)
    plt.axis('off')
    plt.show()


def image_cut_save(path, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    img = Image.open(path)  # 打开图像
    box = (left, upper, right, lower)
    roi = img.crop(box)


    # 保存截取的图片
    roi.save(save_path)


if __name__ == '__main__':
    pic_path = 'D:\\c盘转移\\LES_RNS图像\\s=1\\LES\\kcs-bare-les\\kcs-bare-les-csv1\\XYZ_Internal_Table_table_1.012896e-01.png'
    pic_save_dir_path = 'cut.jpg'
    left, upper, right, lower = 450, 480, 900, 880
    show_cut(pic_path, left, upper, right, lower)
    # image_cut_save(pic_path, left, upper, right, lower, pic_save_dir_path)


    #左下图片：（35，880，545，1280）
    #中间图片：（435，480，915，880）
    #右上图片：（835，50，1315，450）
