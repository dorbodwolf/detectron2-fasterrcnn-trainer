#!/usr/bin/env python
#coding=utf-8
from tools.gaofen_xml_io import GaofenXMLWriter
import os
def saveGaofenXMLFormat(xml_name, img_path, boxes_retrun, scores_return):
    imgFileName = os.path.basename(img_path)  # 图像名称

    writer = GaofenXMLWriter(imgFileName)
    writer.verified = False

    for i in range(boxes_retrun.shape[0]):
        # class_name = category_index[classes_total[i]]['name']
        # Add Chris
        difficult = 0
        writer.addBndBox(int(boxes_retrun[i][0]), int(boxes_retrun[i][1]), int(boxes_retrun[i][2]),
                         int(boxes_retrun[i][3]))
        writer.addScore(scores_return[i])

    writer.save(targetFile=xml_name)