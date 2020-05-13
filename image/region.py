import collections
import numpy as np
import cv2

Point = collections.namedtuple("Point", ["x", "y"])


class Region(object):
    def __init__(self, sub_set=None, value=None):
        self.point_set = sub_set
        self.value = value

    def add(self, point):
        self.point_set.add(point)

    def get_set(self):
        return self.point_set

    def get_value(self):
        return self.value

    def get_size(self):
        return len(self.point_set)

    def __lt__(self, other):
        top = min([point.y for point in self.point_set])
        other_top = min([point.y for point in other.point_set])
        return top < other_top

    def get_border(self):
        """计算边界元素大小"""
        left = min([point.x for point in self.point_set])
        right = max([point.x for point in self.point_set]) + 1

        top = min([point.y for point in self.point_set])
        bottom = max([point.y for point in self.point_set]) + 1

        return left, right, top, bottom

    def generate_image(self):
        left, right, top, bottom = self.get_border()
        width, height = right - left, bottom - top

        image_np = np.zeros((width, height), dtype=np.uint8)

        for point in self.point_set:
            image_np[point.x - left, point.y - top] = 255

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

    def __repr__(self):
        left, right, top, bottom = self.get_border()
        return "{}\t{}\t{}\t{}\t{}".format(left, right, top, bottom, self.value)

