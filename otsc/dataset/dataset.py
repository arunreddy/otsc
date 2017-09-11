# coding=utf-8
from collections import namedtuple

Data = namedtuple('Data', ['X', 'y', 'yp', 'fp', 'L', 'D'])
DataTrainTest = namedtuple('DataTrainTest', ['XL', 'XU', 'yL', 'yU', 'ypL', 'ypU', 'fpL', 'fpU', 'L'])
