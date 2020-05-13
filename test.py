from image_handler import ImageHandler
import cv2


if __name__ == '__main__':

    # 初始化一个类，初始化过程中会请求获取图片
    image_handler = ImageHandler()

    # 判断是否正常获取；
    if image_handler.is_legal():

        # 获取原图像；
        image_handler.save_image("origin.{}".format(image_handler.get_suffix()))

        # 获取处理后的图像；
        processed_image = image_handler.get_gray_static_image()
        cv2.imwrite("processed.png", processed_image)

        # 获取切分的小图像；
        image_list = image_handler.generate_uniform_image()
        for index, image in enumerate(image_list):
            cv2.imwrite("{}.png".format(index), image)

        # 输出预测结果值；
        print(image_handler.get_predict())
