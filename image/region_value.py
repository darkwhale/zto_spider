import numpy as np
import collections
import cv2

from config import standard_avg, standard_std

Point = collections.namedtuple("Point", ["x", "y"])


class RegionValue(object):

    def __init__(self, point_dict=None):
        self.point_dict = point_dict
        self.process_avg_std()

    def get_size(self):
        return len(self.point_dict)

    def process_avg_std(self):
        point_value_list = [value for _, value in self.point_dict.items()]

        avg = np.mean(point_value_list)
        std = np.std(point_value_list)

        for key, value in self.point_dict.items():
            new_value = int((value - avg) * (standard_std / std) + standard_avg)
            new_value = 0 if new_value < 0 else new_value
            new_value = 255 if new_value > 255 else new_value
            self.point_dict[key] = new_value

    def get_border(self):
        left = min([point.x for point in self.point_dict.keys()])
        right = max([point.x for point in self.point_dict.keys()]) + 1

        top = min([point.y for point in self.point_dict.keys()])
        bottom = max([point.y for point in self.point_dict.keys()]) + 1

        return left, right, top, bottom

    def generate_image(self):
        left, right, top, bottom = self.get_border()
        width, height = right - left, bottom - top

        image_np = np.zeros((width, height), dtype=np.uint8)

        for key, value in self.point_dict.items():
            image_np[key.x - left, key.y - top] = self.point_dict[key]

        return self.uniform_piece(image_np)

    def uniform_piece(self, image_piece, length=36):
        left, right, top, bottom = self.get_border()
        width, height = right - left, bottom - top

        remain_height, remain_width = length - height, length - width

        top_fill = remain_height // 2
        bottom_fill = (remain_height + 1) // 2
        left_fill = remain_width // 2
        right_fill = (remain_width + 1) // 2

        return cv2.copyMakeBorder(image_piece, left_fill, right_fill, top_fill, bottom_fill,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def generate_images(self):
        left, right, top, bottom = self.get_border()
        width, height = right - left, bottom - top

        image_np = np.zeros((width, height), dtype=np.uint8)

        for key, value in self.point_dict.items():
            image_np[key.x - left, key.y - top] = self.point_dict[key]

        return self.uniform_pieces(image_np)

    def uniform_pieces(self, image_piece, length=36):
        left, right, top, bottom = self.get_border()
        width, height = right - left, bottom - top

        remain_height, remain_width = length - height, length - width

        top_fill = remain_height // 2
        bottom_fill = (remain_height + 1) // 2
        left_fill = remain_width // 2
        right_fill = (remain_width + 1) // 2

        result_list = []

        left_sum = left_fill + right_fill
        top_sum = top_fill + bottom_fill

        for left_i in range(0, left_fill + 1):
            for top_i in range(0, top_fill + 1):
                right_i = left_sum - left_i
                bottom_i = top_sum - top_i
                result_list.append(cv2.copyMakeBorder(image_piece, left_i, right_i, top_i, bottom_i,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0]))

        return result_list
