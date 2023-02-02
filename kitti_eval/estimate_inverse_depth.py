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
    description="Script for DispNet testing with corresponding groundTruth",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--root-image-dir", default=None, required=True, type=Path, help="path to dataset")
parser.add_argument(
    "--root-save-dir",
    default=None,
    required=True,
    type=Path,
    help="Output directory for saving predictions in a big 3D numpy file",
)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action="store_true", help="no resizing is done")
parser.add_argument("--save-in-local", action="store_true")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_tensor_image(filename: Path, args: argparse.Namespace):
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

    root_image_dir: Path = args.root_image_dir
    root_save_dir: Path = args.root_save_dir
    if args.save_in_local:
        root_save_dir = Path(str(root_save_dir).replace("dugong/", ""))
    root_save_dir.mkdir(parents=True, exist_ok=True)

    for scene_dir in tqdm(root_image_dir.iterdir()):
        save_dir = root_save_dir / scene_dir.name
        save_dir.mkdir(parents=False, exist_ok=True)
        for filepath in tqdm(scene_dir.glob("*.jpg")):
            tgt_img = load_tensor_image(filepath, args)
            pred_disp = disp_net(tgt_img).cpu().numpy()[0, 0]
            save_img_path = root_save_dir / scene_dir.name / f"{filepath.stem}.jpg"
            save_depth_path = root_save_dir / scene_dir.name / f"{filepath.stem}.npy"
            plt.imshow(pred_disp, cmap="plasma"), plt.grid(linestyle=":", linewidth=0.4)
            plt.tight_layout()
            plt.axis("off")
            # plt.show()
            plt.savefig(save_img_path, bbox_inches="tight", pad_inches=0)
            np.save(save_depth_path, pred_disp)


if __name__ == "__main__":
    main()
