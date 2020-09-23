#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from detectron2.data.datasets import register_coco_instances
# register_coco_instances("bridge_dataset_train", {}, "/home/asd/Mission/GaoFen/bridge_new/data/data_aug_coco/coco/bridge_train_cocostyle.json", "/home/asd/Mission/GaoFen/bridge_new/data/data_aug_coco/coco/bridge_train")
# register_coco_instances("bridge_dataset_test", {}, "/home/asd/Mission/GaoFen/bridge_new/data/test_data_coco/bridge_test_cocostyle.json", "/home/asd/Mission/GaoFen/bridge_new/data/test_data_coco/bridge_test")


logger = logging.getLogger("detectron2")

from contextlib import contextmanager

from detectron2.utils.comm import get_world_size, is_main_process

from tqdm import tqdm

from PIL import Image

import numpy as np
import glob
import os.path as osp

import argparse

from detectron2.engine.defaults import DefaultPredictor

from tools.GaofenXMLFormat import saveGaofenXMLFormat

#from ensemble_boxes import weighted_boxes_fusion

from detectron2.utils.visualizer import ColorMode, Visualizer

import cv2

from osgeo import gdal, ogr, osr
import geopandas as gpd
from shapely.geometry import Polygon

import pandas as pd

import json

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)



# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.MODEL.DEVICE = 'cuda'
#     cfg.MODEL.WEIGHTS = '/home/asd/Project/BridgeDetection/Faster-RCNN-Trainer/output/model_final.pth'
#     cfg.freeze()
#     default_setup(
#         cfg, args
#     )  # if you don't like any of the default setup, write your own setup code
#     return cfg


def select_top(boxes, scores, prediction, thresholds_for_classes):
    isValidScaler = torch.zeros(len(boxes), dtype=torch.bool)
    for i, box in enumerate(boxes):
        if prediction['scores'][i] > thresholds_for_classes:
            isValidScaler[i] = True
    top_boxes = boxes[isValidScaler].tensor.numpy()
    top_scores = scores[isValidScaler].numpy()
    return top_boxes, top_scores

def wbf(im, boxes, scores, classes):
    """
    weighted boxes fusion
    """
    w, h = im.size
    boxes[:,0] /= w
    boxes[:,2] /= w
    boxes[:,1] /= h
    boxes[:,3] /= h

    boxes_list = boxes.tolist()
    scores_list = scores.tolist()
    labels_list = classes.tolist()
    
    boxes, scores, _ = weighted_boxes_fusion([boxes_list], [scores_list], [labels_list], weights=None, iou_thr=0.2)

    boxes[:,0] *= w
    boxes[:,2] *= w
    boxes[:,1] *= h
    boxes[:,3] *= h
    return boxes, scores



def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            # print x,y
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords


def __hash__(df):
    geom_hash = [hash(tuple(geom.coords)) for geom in df.geometry]
    df['geom_hash'] = geom_hash
    return df

def group_testimages(imgs):
    """
    对测试图像按照空间位置分组
    """
    df = gpd.GeoDataFrame(columns=['imgid','geometry'])
    # df = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])       
    
    for img in tqdm(imgs):
        ds = gdal.Open(img)
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        ext=GetExtent(gt,cols,rows)

        src_srs=osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection())
        #tgt_srs=osr.SpatialReference()
        #tgt_srs.ImportFromEPSG(4326)
        tgt_srs = src_srs.CloneGeogCS()
        geo_ext=ReprojectCoords(ext,src_srs,tgt_srs)
        geo_ext_arr = np.array(geo_ext, dtype=float)
        lat_list_arr = geo_ext_arr[:, 1]
        lon_list_arr = geo_ext_arr[:, 0]
        lat_list = lat_list_arr.tolist()
        lon_list = lon_list_arr.tolist()
        polygon_geom = Polygon(zip(lon_list, lat_list))
        crs = {'init': 'epsg:4326'}
        # print(polygon.geometry)
        df = df.append({'imgid': img, 'geometry':polygon_geom}, ignore_index=True)
    # df = __hash__(df)
    df['WKT'] = df['geometry'].apply(lambda x: str(x))
    gdf = df.groupby(['WKT'])
    # grouped_imgs = gdf["imgid"].agg(lambda column: ["".join(column)])
    grouped_imgs = gdf["imgid"].apply(list)
    # grouped_imgs = gdf["imgid"].agg(lambda column: colume, [])

    return grouped_imgs

def generalInference(model, im_names, predictor):
    # num_devices = get_world_size()
    with inference_context(model), torch.no_grad():
        for im_name in tqdm(im_names):
            im=Image.open(im_name)
            w, h = im.size
            # imSize = (h, w)
            # mode = im.mode
            im_data = np.array(im) 
            # Convert RGB to BGR 
            open_cv_image = im_data[:, :, ::-1].copy()

            xml_name=os.path.join(save_dir,os.path.splitext(osp.basename(im_name))[0]+".xml")
            if os.path.exists(xml_name):
                print("exists:",xml_name)
            else:
                output = predictor(open_cv_image)
                # v = Visualizer(im_data, scale=0.5)
                # out = v.draw_instance_predictions(output["instances"].to("cpu"))
                # cv2.imshow("show_instances", out.get_image()[:, :, ::-1])
                if "instances" in output:
                    instances = output["instances"]#.to(self.cpu_device)
                    prediction = instances.get_fields()
                    boxes = prediction['pred_boxes']
                    scores = prediction['scores']
                    classes = prediction['pred_classes']
                    # select_top() 选出top预测结果
                    boxes = boxes.tensor.cpu().numpy()
                    scores = scores.cpu().numpy()
                    classes = classes.cpu().numpy()

                    # wbf() weighted boxes fusion调用
                    
                    saveGaofenXMLFormat(xml_name, im_name, boxes, scores)

def xyxy_to_xywh(box):
    """
    boxes表示方法转换
    """
    #print(box)\
    import copy
    arr = copy.deepcopy(box)
    #print(arr)
    # from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
    arr[2] -= arr[0]
    arr[3] -= arr[1]
    
    return abs(arr)

def groupedInference(img_groups, model, predictor):
    coco_list = []
    with inference_context(model), torch.no_grad():
        for names in tqdm(img_groups):
            # print(names)
            group_results = pd.DataFrame(columns=["img", "boxes", "scores", "mean_score", "classes", "count"])
            for im_name in names:
                # print(im_name)
                im=Image.open(im_name)
                # w, h = im.size
                im_data = np.array(im) 
                # Convert RGB to BGR 
                open_cv_image = im_data[:, :, ::-1].copy()
                output = predictor(open_cv_image)
                #v = Visualizer(im_data, scale=0.5)
                #out = v.draw_instance_predictions(output["instances"].to("cpu"))
                #cv2.namedWindow("ins", cv2.WINDOW_NORMAL)
                #cv2.imshow("ins", out.get_image()[:, :, ::-1])
                #cv2.waitKey (0)
                if "instances" in output:
                    instances = output["instances"]#.to(self.cpu_device)
                    prediction = instances.get_fields()
                    boxes = prediction['pred_boxes']
                    scores = prediction['scores']
                    classes = prediction['pred_classes']
                    # select_top() 选出top预测结果
                    boxes = boxes.tensor.cpu().numpy()
                    #print(boxes)
                    scores = scores.cpu().numpy()
                    classes = classes.cpu().numpy()
                    if len(boxes) == 0:
                        xml_name=os.path.join(save_dir,os.path.splitext(osp.basename(im_name))[0]+".xml")
                        if os.path.exists(xml_name):
                            pass
                            # print("exists:",xml_name)
                        else:   
                            saveGaofenXMLFormat(xml_name, im_name, boxes, scores)
                    else:
                        group_results = group_results.append({"img": im_name,"boxes":boxes, "scores":scores,"mean_score":scores.mean(), "classes":classes, "count":len(classes)}, ignore_index=True)
                    # group_results = group_results.append({"img": im_name,"boxes":boxes, "scores":scores,"mean_score":scores.mean(), "classes":classes, "count":len(classes)}, ignore_index=True)
            if len(group_results) == 0:
                pass
            elif len(group_results) == 1:
                for i in range(len(group_results.iloc[0]['scores'])):
                    #print(group_results.iloc[0]['boxes'])
                    box_xywh = xyxy_to_xywh(group_results.iloc[0]['boxes'][i,:])
                    coco_list.append({"image_id":int(os.path.basename(names[0])[:-4]), "category_id":group_results.iloc[0]['classes'][i],
                         "score":group_results.iloc[0]['scores'][i], "bbox":box_xywh.tolist()})
                xml_name=os.path.join(save_dir,os.path.splitext(osp.basename(names[0]))[0]+".xml")
                if os.path.exists(xml_name):
                    pass
                    # print("exists:",xml_name)
                else:
                    saveGaofenXMLFormat(xml_name, names[0], group_results.iloc[0]['boxes'],  group_results.iloc[0]['scores'])
            else:
                # print(group_results["mean_score"].mean())
                if group_results["mean_score"].mean() < 0.8:
                    top = group_results.sort_values(['mean_score'],ascending=False).iloc[0]
                    #print(top['boxes'])
                    #print("\nshowtim.............................e\n")
                    for _, result in group_results.iterrows():
                        im_name = result['img']
                        boxes = top['boxes']
                        #print(boxes)
                        scores = top['scores']
                        classes = top['classes']
                        #print(im_name)
                        for i in range(len(scores)):
                            #print(boxes[i,:])
                            box_xywh = xyxy_to_xywh(boxes[i,:])
                            #print(box_xywh)
                            coco_list.append({"image_id":int(os.path.basename(im_name)[:-4]), "category_id":classes[i],
                                "score":scores[i], "bbox":box_xywh.tolist()})
                        xml_name=os.path.join(save_dir,os.path.splitext(osp.basename(im_name))[0]+".xml")
                        if os.path.exists(xml_name):
                            pass
                            # print("exists:",xml_name)
                        else:   
                            saveGaofenXMLFormat(xml_name, im_name, boxes, scores)
                else:
                    top = group_results[group_results['mean_score'] >= 0.8].sort_values(['count'],ascending=False).iloc[0]
                    # group_results.sort_values(['mean_score'],ascending=False)
                    # print(above_80)
                    for _, result in group_results.iterrows():
                        im_name = result['img']
                        boxes = top['boxes']
                        scores = top['scores']
                        classes = top['classes']
                        for i in range(len(scores)):
                            box_xywh = xyxy_to_xywh(boxes[i,:])
                            coco_list.append({"image_id":int(os.path.basename(im_name)[:-4]), 
                                    "category_id":classes[i],"score":scores[i], "bbox":box_xywh.tolist()})
                        xml_name=os.path.join(save_dir,os.path.splitext(osp.basename(im_name))[0]+".xml")
                        if os.path.exists(xml_name):
                            pass
                            # print("exists:",xml_name)
                        else:   
                            saveGaofenXMLFormat(xml_name, im_name, boxes, scores)
    #print(coco_list)
    # with open('data.json', 'w') as f:
    #     json.dump(coco_list[0], f)

def setup(args, weight):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = weight
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args, weight):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cfg = setup(args, weight)
    model = build_model(cfg)
    # logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    predictor = DefaultPredictor(cfg)
    # thresholds_for_classes = 0.7
    im_names = glob.glob(osp.join(images_dir, '*.tif'))

    img_groups =  group_testimages(im_names)
    groupedInference(img_groups, model, predictor)
    # generalInference(model, im_names, predictor)
    return

    

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # num_machines=args.num_machines
    # machine_rank=args.machine_rank
    # dist_url=args.dist_url
    ioParser = argparse.ArgumentParser()
    ioParser.add_argument('--in_path', type=str,help='输入图片路径', default=r"/input_path")
    ioParser.add_argument('--out_path', type=str,help='输出xml路径', default=r"/output_path")
    # ioParser.add_argument('--config_file', type=str,help='输出xml路径', default=r"configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
    config = ioParser.parse_args()
    images_dir = config.in_path
    save_dir = config.out_path
    # args.config_file = config.config_file
    args.config_file = "./configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"
    weight = "./output/model_0069999.pth"
    main(args, weight)
