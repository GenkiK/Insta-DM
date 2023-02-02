import argparse
import datetime
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from imageio.v2 import imread
from skimage.transform import resize as imresize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Script for DispNet testing with corresponding groundTruth",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--img-path", default=None, required=True, type=Path, help="path to image")
parser.add_argument(
    "--output_dir",
    default=None,
    required=True,
    type=Path,
    help="Output directory for saving depth map",
)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action="store_true", help="no resizing is done")
parser.add_argument("--save-in-local", action="store_true")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
    return tensor_img


@torch.no_grad()
def main():
    args = parser.parse_args()
    print("=> Tested at {}".format(datetime.datetime.now().strftime("%m-%d-%H:%M")))
    print("=> Load dispnet model from {}".format(args.pretrained_dispnet))

    sys.path.insert(1, os.path.join(sys.path[0], ".."))
    import models

    disp_net = models.DispResNet().to(device)
    disp_weights = torch.load(args.pretrained_dispnet, map_location=device)
    disp_net.load_state_dict(disp_weights["state_dict"])
    disp_net.eval()

    output_dir: Path = args.output_dir
    if args.save_in_local:
        output_dir = Path(str(output_dir).replace("dugong/", ""))

    if Path(args.img_path).suffix[1:].lower() in {"jpg", "jpeg", "png"}:
        img_paths = [args.img_path]
    else:
        # img_paths = args.img_path.glob("*.jpg")
        img_paths = args.img_path.glob("*.png")
    for img_path in img_paths:
        tgt_img = load_tensor_image(img_path, args)
        pred_disp = disp_net(tgt_img).cpu().numpy()[0, 0]

        # save_path = output_dir / test_files[i].replace("/data", "")
        save_path = output_dir / str(img_path).replace(str(img_path.parents[1]), "")[1:]
        print(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 3))
        ax.imshow(pred_disp[:, 50:-50], cmap="plasma")
        ax.axis("off")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    main()
