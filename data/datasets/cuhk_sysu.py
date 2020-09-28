import os.path as osp

import numpy as np
from numpy.core.records import record
from scipy.io import loadmat


class CUHKSYSU:
    def __init__(self, root):
        train = CUHK_SYSU(root=root, mode="train")
        test = CUHK_SYSU(root=root, mode="test")
        data_path = osp.join(root, "Image/SSM")

        def f(dataset):
            res = []
            for img, record in zip(dataset.imgs, dataset.records):
                img_path = osp.join(data_path, img)
                for box, pid in zip(record["boxes"], record["gt_pids"]):
                    if pid != -1:
                        # set camera id to -1
                        res.append((img_path, box, pid, -1))
            return res

        self.train = f(train)
        self.test = f(test)

    @property
    def num_train_pids(self):
        pids = [info[2] for info in self.train]
        return len(set(pids))


class CUHK_SYSU:
    def __init__(self, root, mode):
        self.root = root
        self.data_path = osp.join(self.root, "Image", "SSM")
        self.mode = mode
        assert self.mode in ("train", "test")
        self.imgs = self._load_image_set_index()
        self.records = self.gt_roidb()

    def gt_roidb(self):
        # Load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        for im_name, _, boxes in all_imgs:
            im_name = str(im_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(im_name)
            boxes = boxes[valid_index]
            name_to_boxes[im_name] = boxes.astype(np.int32)
            name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)

        def _set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return
            print("Warning: person {} box {} cannot find in Images".format(pid, box))

        # Load all the train/probe/test persons and number their pids from 0 to N-1
        # Background people have pid == -1
        if self.mode == "train":
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for im_name, box, __ in scenes:
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)
        else:
            test = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            test = test["TestG50"].squeeze()
            for index, item in enumerate(test):
                # query
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                _set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, __ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index)

        # Construct the gt_roidb
        gt_roidb = []
        for im_name in self.imgs:
            boxes = name_to_boxes[im_name]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]  # (x1, y1, x2, y2)
            pids = name_to_pids[im_name]
            gt_roidb.append({"im_name": im_name, "boxes": boxes, "gt_pids": pids, "flipped": False})

        return gt_roidb

    def _load_image_set_index(self):
        """
        Load the indexes for the specific subset (train / test).
        For PSDB, the index is just the image file name.
        """
        # test pool
        test = loadmat(osp.join(self.root, "annotation", "pool.mat"))
        test = test["pool"].squeeze()
        test = [str(a[0]) for a in test]
        if self.mode in ("test", "probe"):
            return test
        # all images
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training
        return list(set(all_imgs) - set(test))
