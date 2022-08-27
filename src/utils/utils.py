import os

import numpy as np
from PIL import Image, ImageCms

import torch
import torchvision.utils as vutils


srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")
LAB2RGB = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")


def load_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples


def write_image(image, path, mode="img", c_space="RGB"):
    if mode == "img":
        img = image * 255
        img = img.transpose(1, 2, 0)
        img = img.astype(np.uint8)
        h, w, ch = img.shape
        if c_space == "RGB":
            if ch == 1:
                img = img.reshape((h, w))
                result = Image.fromarray(img)
                result.save(path + ".jpg")
            elif ch == 4:
                img_gray = img[:, :, 3]
                img_color = img[:, :, :3]
                result_gray = Image.fromarray(img_gray)
                result_color = Image.fromarray(img_color)
                result_gray.save(path + "_gray.jpg")
                result_color.save(path + "_color.jpg")
            else:
                result = Image.fromarray(img)
                result.save(path + ".jpg")
        elif c_space == "LAB":
            tmp = Image.fromarray(img, mode="LAB")
            result = ImageCms.applyTransform(tmp, LAB2RGB)
            result.save(path + ".jpg")
        else:
            result = Image.fromarray(img, mode=c_space).convert("RGB")
            result.save(path + ".jpg")
    else:
        np.save(path + ".npy", image)


def write_outputs(writer, outputs, count, prefix=""):
    for k, v in outputs.items():
        for i, vv in enumerate(v):
            if isinstance(vv, torch.Tensor):
                if len(vv.shape) == 3:
                    vv = vv.unsqueeze(axis=1)
                elif len(vv.shape) == 4:
                    vv = vv.reshape((-1, vv.shape[2], vv.shape[3]))
                    vv = vv.unsqueeze(axis=1)
                x = vutils.make_grid(vv, normalize=True, scale_each=True)
                if prefix == "":
                    writer.add_image(k + "_time{}".format(i), x, count)
                else:
                    writer.add_image(prefix + "/" + k + "_time{}".format(i), x, count)
