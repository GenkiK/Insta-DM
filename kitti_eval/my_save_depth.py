import argparse
import datetime
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio import imread
from skimage.transform import resize as imresize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Script for DispNet testing with corresponding groundTruth", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--data_dir", default=None, required=True, type=Path, help="path to dataset")
parser.add_argument("--output_dir", default=None, required=True, type=Path, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action="store_true", help="no resizing is done")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
    return tensor_img, img


@torch.no_grad()
def main():
    args = parser.parse_args()
    print("=> Tested at {}".format(datetime.datetime.now().strftime("%m-%d-%H:%M")))
    print("=> Load dispnet model from {}".format(args.pretrained_dispnet))

    sys.path.insert(1, os.path.join(sys.path[0], ".."))
    import models

    disp_net = models.DispResNet().to(device)
    disp_weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(disp_weights["state_dict"])
    disp_net.eval()

    data_dir = args.data_dir
    with open(args.dataset_list, "r") as f:
        test_files = list(f.read().splitlines())
    print("=> {} files to test".format(len(test_files)))

    # output_dir: Path = args.output_dir
    # output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(test_files))):
        tgt_img, ori_img = load_tensor_image(data_dir / test_files[i], args)
        # tgt_img = load_tensor_image(data_dir / test_files[i], args)
        pred_disp = disp_net(tgt_img).cpu().numpy()[0, 0]

        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(ori_img.transpose(1, 2, 0) / 255, vmin=0, vmax=1), plt.grid(linestyle=":", linewidth=0.4)
        fig.add_subplot(2, 1, 2)
        plt.imshow(pred_disp), plt.grid(linestyle=":", linewidth=0.4)
        fig.tight_layout(), plt.show()

    #     if i == 0:
    #         disp_predictions = np.zeros((len(test_files), *pred_disp.shape))
    #     disp_predictions[i] = 1 / pred_disp
    # np.save(output_dir / "disp_predictions.npy", disp_predictions)


if __name__ == "__main__":
    main()
