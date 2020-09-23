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
from detectron2.data.dataset_mapper import DatasetMapper

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

import numpy as np

from detectron2.data.datasets import register_coco_instances
register_coco_instances("bridge_dataset_train", {}, "/home/asd/Mission/GaoFen/bridge_new/data/data_wenhe_gai/aug/train/coco/bridge_train_cocostyle.json", "/home/asd/Mission/GaoFen/bridge_new/data/data_wenhe_gai/aug/train/coco/bridge_train")
register_coco_instances("208_test", {}, "/home/asd/Mission/GaoFen/bridge_new/data/data_aug_coco/test_data_coco/bridge_test_cocostyle.json", "/home/asd/Mission/GaoFen/bridge_new/data/data_aug_coco/test_data_coco/bridge_test")
register_coco_instances("laji_test", {}, "/home/asd/Mission/GaoFen/bridge_new/data/data_wenhe_gai/aug/test/coco/bridge_test_cocostyle.json", "/home/asd/Mission/GaoFen/bridge_new/data/data_wenhe_gai/aug/test/coco/bridge_test")

logger = logging.getLogger("桥梁训练器")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def _get_val_loss(data, model):
    """
    返回lossdict
    """
    # with inference_context(model), torch.no_grad():
    model.train()
    loss_dict = model(data)
    # metrics_dict = {
    #     k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
    #     for k, v in metrics_dict.items()
    # }
    # total_losses_reduced = sum(loss for loss in metrics_dict.values())

    # print(loss_dict)
    # loss_dict = model(data)
    # losses = sum(loss_dict.values())
    # assert torch.isfinite(losses).all(), loss_dict

    loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}    
    return loss_dict_reduced

def do_loss_eval(cfg, storage, model, test_data_loaders):
    """
    计算验证集的loss，并存储在EventStorage对象中
    """        
    losses = []
    for data_loader_dict in test_data_loaders:
        dataset_name =  data_loader_dict["name"]
        data_loader = data_loader_dict["data_loader"]
        for idx, inputs in enumerate(data_loader):            
            loss_batch_dict = _get_val_loss(inputs, model)
            losses_batch = sum(loss_batch_dict.values())
            assert np.isfinite(losses_batch).all(), loss_batch_dict
            losses.append(losses_batch)
        mean_loss = np.mean(np.array(losses, dtype=float))
        storage.put_scalar('val_loss_{}'.format(dataset_name), mean_loss)
        storage.put_scalar("time", 10)
        comm.synchronize()
        # return losses

def do_train(cfg, model, resume=False):
    # 模型设置训练模式
    model.train()
    # 构建优化器
    optimizer = build_optimizer(cfg, model)
    # 构建学习率调整策略
    scheduler = build_lr_scheduler(cfg, optimizer)

    # 断点管理对象
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    # 可用于恢复训练的起始训练步
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    # 最大迭代次数
    max_iter = cfg.SOLVER.MAX_ITER

    # 这里的PeriodicCheckpointer是fvcore.common.checkpoint中的类，可以用于在指定checkpoint处保存和加载模型
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter), # 负责终端loss登信息的打印
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # 构建batched训练data loader
    data_loader = build_detection_train_loader(cfg)
    # 构建用于获取测试loss的 test data loader
    test_data_loaders = []
    for dataset_name in cfg.DATASETS.TEST:
        test_data_loaders.append({"name": dataset_name, "data_loader": build_detection_test_loader(cfg, dataset_name, DatasetMapper(cfg,True))})
    logger.info("从第{}轮开始训练".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            # 每个迭代的开始调用，更新storage对象的游标
            storage.step()

            loss_dict = model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                # 将该轮前向传播的loss放入storage对象的容器中（storage.histories()，后面读取该容器来打印终端）
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # 将该轮学习率放入storage对象的容器中
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
            
            # if iteration % 21 == 0:
            #     do_loss_eval(cfg, storage, model, test_data_loaders)
            #     for writer in writers:
            #         writer.write()
            
            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                do_loss_eval(cfg, storage, model, test_data_loaders)
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.config_file = '/home/asd/Project/BridgeDetection/Faster-RCNN-Trainer/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml'
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
