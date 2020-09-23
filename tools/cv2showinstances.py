#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from ensemble_boxes import weighted_boxes_fusion

from detectron2.utils.visualizer import ColorMode, Visualizer

import cv2

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



def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file("/home/asd/Project/BridgeDetection/Faster-RCNN-Trainer/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = '/home/asd/Project/BridgeDetection/Faster-RCNN-Trainer/output/model_final.pth'
    cfg.freeze()
    return cfg


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cfg = setup()
    model = build_model(cfg)
    # logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS
    )

    predictor = DefaultPredictor(cfg)
    # thresholds_for_classes = 0.7
    im_names = glob.glob(osp.join(images_dir, '*.tif'))
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
            output = predictor(open_cv_image)
            v = Visualizer(im_data, scale=0.5)
            out = v.draw_instance_predictions(output["instances"].to("cpu"))
            cv2.namedWindow("ins")
            cv2.imshow("ins", out.get_image()[:, :, ::-1])
            cv2.waitKey (1000)
            # cv2.destroyAllWindows()


if __name__ == "__main__":
    # num_machines=args.num_machines
    # machine_rank=args.machine_rank
    # dist_url=args.dist_url
    images_dir = "/home/asd/Mission/GaoFen/bridge_new/data/data_wenhe_gai/aug/test/coco/bridge_test"
    main()


# In[ ]:





# In[ ]:




