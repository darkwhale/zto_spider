import requests
import json
import base64
import os
from PIL import Image, ImageSequence
import numpy as np
import cv2
from functools import reduce
from io import BytesIO
from collections import defaultdict

from config import *
from exception import RequestException
from region import Region, Point
from region_value import RegionValue
from predict import predict


def get_url():
    try_num = 5
    while try_num:
        try:
            content = requests.post(zto_image_url, timeout=3)
            if content.status_code == 200:
                return content
        finally:
            try_num -= 1

    raise RequestException("request异常")


class ImageHandler(object):
    def __init__(self):

        self.status = None

        # noinspection PyBroadException
        try:
            self.image_content = get_url()

            self.origin_image_dict = self.image_content.text

            base_dict = json.loads(self.origin_image_dict)
            self.image_base64_str = base_dict.get("result").get("image")
            self.id = base_dict.get("result").get("id")

            base64_index = self.image_base64_str.find("base64")
            self.fore_info, self.image_info = self.image_base64_str[: base64_index], \
                                              self.image_base64_str[base64_index + 7:]

            self.status = 0
        except RequestException:
            self.status = 1
        except Exception:
            self.status = 2

    def is_legal(self):
        return self.status == 0

    def get_origin_content(self):
        return self.origin_image_dict

    def get_suffix(self):
        return "png" if "png" in self.fore_info else "gif"

    def get_fore_info(self):
        return self.fore_info

    def get_image_info(self):
        return self.image_info

    def get_id(self):
        return self.id

    def save_image(self, image_path):
        with open(image_path, "wb") as writer:
            writer.write(base64.b64decode(self.image_info))

    def get_gray_static_image(self):
        # 1.读取图像，2.判断为png还是gif，3.处理返回
        # 先保存图片

        image_base64 = base64.b64decode(self.image_info)
        image_io = BytesIO(image_base64)
        image = Image.open(image_io)

        # png直接返回灰度图像，gif读取帧然后合并图像
        if self.get_suffix() == "png":
            np_list = np.asarray(image)
            np_merge = 255 - cv2.cvtColor(np_list, cv2.COLOR_BGR2GRAY)

        else:
            np_list = list()
            for index, image_frame in enumerate(ImageSequence.Iterator(image)):
                np_list.append(np.asarray(image_frame))

            np_merge = reduce(np.minimum, np_list)

        return np_merge

    def generate_uniform_image(self):
        """获取标准化的图像块"""
        image = self.get_gray_static_image()
        # _, image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY_INV)
        # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)

        sub_region_vertical = region_vertical if self.get_suffix() == "gif" else region_png_vertical
        sub_region_part1 = region_part1 if self.get_suffix() == "gif" else region_png_part1
        sub_region_part2 = region_part2 if self.get_suffix() == "gif" else region_png_part2
        sub_region_part3 = region_part3 if self.get_suffix() == "gif" else region_png_part3
        sub_region_part4 = region_part4 if self.get_suffix() == "gif" else region_png_part4

        image_part1 = image[sub_region_vertical[0]: sub_region_vertical[1], sub_region_part1[0]: sub_region_part1[1]]
        image_part2 = image[sub_region_vertical[0]: sub_region_vertical[1], sub_region_part2[0]: sub_region_part2[1]]
        image_part3 = image[sub_region_vertical[0]: sub_region_vertical[1], sub_region_part3[0]: sub_region_part3[1]]
        image_part4 = image[sub_region_vertical[0]: sub_region_vertical[1], sub_region_part4[0]: sub_region_part4[1]]

        image_part_list = [image_part1, image_part2, image_part3, image_part4]

        if self.get_suffix() == "png":
            image_list = [self.normal_region_png(image_part) for image_part in image_part_list]
        else:
            image_list = [self.normal_region(image_part) for image_part in image_part_list]

        return image_list

    def get_predict(self):
        return predict(self.generate_uniform_image())

    @staticmethod
    def normal_region_png(image):
        width, height = image.shape

        point_dict = dict()

        for i in range(width):
            for j in range(height):
                if image[i, j] > 10:
                    point_dict[Point(i, j)] = image[i, j]

        region_value = RegionValue(point_dict)
        return region_value.generate_image()

    @staticmethod
    def normal_region(image):
        width, height = image.shape

        fill_dict = defaultdict(set)
        for i in range(width):
            for j in range(height):
                if image[i, j] < 240:
                    fill_dict[image[i, j]].add(Point(i, j))

        maximum_value = max(fill_dict, key=lambda x: len(fill_dict.get(x)))
        maximum_set = fill_dict.get(maximum_value)

        region = Region(maximum_set, maximum_value)

        return region.generate_image()
