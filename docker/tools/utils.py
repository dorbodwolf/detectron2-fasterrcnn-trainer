# encoding: utf-8
"""
@version: 3.6
@author: mas
@file: utils.py
@time: 2020/2/25 11:11
"""
import os
import os.path as osp
from osgeo import gdal,ogr,osr
from collections import defaultdict
import json
import argparse
import time
import glob
import multiprocessing
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon,PolygonsOnImage
from tools.pascal_voc_io import *
import numpy as np
from skimage import io
from tqdm import tqdm
from pathlib import Path

