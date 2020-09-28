# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir
import numpy as np

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
import pickle as pk
import torchvision.transforms as T
from data.datasets.dataset_loader import read_image
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def pickle(data, file_path):
    with open(file_path, "wb") as f:
        pk.dump(data, f, pk.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, "rb") as f:
        data = pk.load(f)
    return data


def get_model_transform():
    args = unpickle("test_args.pkl")

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param("output/resnet50_model_80.pth")
    model.eval()
    model.cuda()

    transform = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST),
                           T.ToTensor(),
                           T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])
    return model, transform


if __name__ == '__main__':
    # main()
    model, transform = get_model_transform()
    recalls = joblib.load("two-stage-data/recalls.pkl")
    y_scores = joblib.load("two-stage-data/y_scores.pkl")
    y_trues = joblib.load("two-stage-data/y_trues.pkl")
    aps = []
    accs = []
    topk = [1, 5, 10]
    nu = 0
    for recall, y_score, y_true in tqdm(zip(recalls, y_scores, y_trues)):
        sims = []
        k = 10
        for i in range(k):
            img1 = transform(read_image("two-stage-data/top10_imgs/query-%s.jpg" % nu)).unsqueeze(0).cuda()
            feat1 = torch.nn.functional.normalize(model(img1), dim=1, p=2)
            img2 = transform(read_image("two-stage-data/top10_imgs/gallery-%s-%s.jpg" % (nu, i))).unsqueeze(0).cuda()
            feat2 = torch.nn.functional.normalize(model(img2), dim=1, p=2)
            sim = float((feat1 * feat2).sum())
            sims.append(sim)
        sims = np.array(sims)
        inds = sims.argsort()[::-1]
        # y_true[0], y_true[ind] = y_true[ind], y_true[0]
        y_true[:k] = y_true[:k][inds]
        nu += 1
        ap = 0 if recall == 0 else average_precision_score(y_true, y_score) * recall
        aps.append(ap)
        accs.append([min(1, sum(y_true[:k])) for k in topk])
        # if nu == 15:
        #     break
    print('  mAP = {:.2%}'.format(np.mean(aps)))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print('  top-{:2d} = {:.2%}'.format(k, accs[i]))
