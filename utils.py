import math

import cv2 as cv
import numpy


class Color:
    BLUE, GREEN = range(2)


class Line:
    def __init__(self, image, lineType):
        # Only use the corresponding color channel
        image = image[:,:,lineType]
        # cv.HoughLines only works with binary images,
        # so we have to apply a threshold
        _, image = cv.threshold(image, 0, 255, cv.THRESH_OTSU)
        # Find the line using probabilistic Hough transform
        # Precision: 1px, 1 rad
        # Minimum vote count 200
        lines = cv.HoughLinesP(image, 1, 1 * numpy.pi / 180, 200, minLineLength=200, maxLineGap=10)
        # Hough transform finds multiple lines
        # because the original line is too thick,
        # so we just take the first one
        self.x1, self.y1, self.x2, self.y2 = lines[0,0]

    def startPoint(self):
        return self.x1, self.y1

    def endPoint(self):
        return self.x2, self.y2

    def length(self, v):
        x, y = v
        return math.sqrt(x * x + y * y)

    def distance(self, p0, p1):
        return self.length(self.vector(p0, p1))

    def vector(self, b, e):
        x, y = b
        X, Y = e
        return (X - x, Y - y)

    def dot(self, v, w):
        x, y = v
        X, Y = w
        return x * X + y * Y

    def unit(self, v):
        x, y = v
        mag = self.length(v)
        return (x / mag, y / mag)

    def scale(self, v, sc):
        x, y = v
        return (x * sc, y * sc)

    def add(self, v, w):
        x, y = v
        X, Y = w
        return (x + X, y + Y)

    def pointDistance(self, point):
        start = self.startPoint()
        end = self.endPoint()

        line_vec = self.vector(start, end)
        pnt_vec = self.vector(start, point)
        line_len = self.length(line_vec)
        line_unitvec = self.unit(line_vec)
        pnt_vec_scaled = self.scale(pnt_vec, 1.0 / line_len)
        t = self.dot(line_unitvec, pnt_vec_scaled)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nearest = self.scale(line_vec, t)
        return self.distance(nearest, pnt_vec)


class Digit:
    def __init__(self, bbox, image):
        self.bbox = bbox
        self.image = image
        self.greenCounted = False
        self.blueCounted = False

    def distance(self, bbox):
        (x, y, _, _) = self.bbox
        (x1, y1, _, _) = bbox
        return math.sqrt((x1 - x)**2 + (y1 - y)**2)


def closestDigit(digits, bbox):
    if len(digits) < 1:
        return 0, None
    ret = min(digits, key=lambda d: d.distance(bbox))
    if ret.distance(bbox) < 20:
        return 1, ret
    else:
        return 0, ret

