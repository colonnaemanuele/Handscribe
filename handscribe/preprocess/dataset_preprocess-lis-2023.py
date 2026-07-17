import re
import os
import cv2
import pdb
import glob
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


def txt2dict(anno_path, dataset_type):
    """
    Parse a pipe-separated .txt annotation file (header + rows).
    Expected columns (example header): id_clip|id_sentence|name|signer|domain|translation|...
    We extract:
      - name -> used as fileid and folder name
      - signer -> signer
      - translation -> label
    """
    print(f"Generate information dict from {anno_path}")
    info_dict = dict()
    info_dict['prefix'] = anno_path.rsplit("/", 3)[0] + "/dataset_split/features"

    # Read lines and skip header
    with open(anno_path, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

    if not lines:
        return info_dict

    # Assume first non-empty line is header if it contains '|' and column names
    header = lines[0] if '|' in lines[0] else None
    data_lines = lines[1:] if header is not None else lines

    for file_idx, line in tqdm(enumerate(data_lines), total=len(data_lines)):
        parts = line.split("|")
        if len(parts) < 6:
            # skip malformed lines
            continue
        name = parts[2].strip()
        signer = parts[3].strip()
        translation = parts[5].strip()
        # Count frames inside the folder (images)
        img_folder = os.path.join(info_dict['prefix'], name)
        num_frames = len(glob.glob(os.path.join(img_folder, "*")))
        info_dict[file_idx] = {
            'fileid': name,
            'folder': f"{dataset_type}/{name}",
            'signer': signer,
            'label': translation,
            'num_frames': num_frames,
            'original_info': line,
        }
    return info_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")


def resize_img(img_path, dsize='210x260px'):
    dsize = tuple(int(res) for res in re.findall(r"\d+", dsize))
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_dataset(video_idx, dsize, info_dict):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        rs_img_path = img_path.replace("210x260px", dsize)
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='GFSlowFastSign/preprocess/dataset_LIS_v2', help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='GFSlowFastSign/dataset/dataset_LIS_v2/dataset_split', help='path to the dataset')
    parser.add_argument('--annotation-prefix', type=str, default='manual/dataset_LIS_v2_{}.txt', help='annotation prefix')
    parser.add_argument('--output-res', type=str, default='256x256px', help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true', help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true', help='whether adopts multiprocessing to accelate the preprocess')

    args = parser.parse_args()
    mode = ["dev", "test", "train"]
    if not os.path.exists(f"./{args.dataset}"): 
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = txt2dict(f"{args.dataset_root}/{args.annotation_prefix.format(md)}", dataset_type=md)
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # generate groudtruth stm for evaluation
        generate_gt_stm(information, f"{args.dataset}/{args.dataset.split("/",2)[2]}-groundtruth-{md}.stm")
        # resize images
        # if args.process_image:
        #     video_index = np.arange(len(information) - 1)
        #     print(f"Resize image to {args.output_res}")
        #     if args.multiprocessing:
        #         run_mp_cmd(10, partial(resize_dataset, dsize=args.output_res, info_dict=information), video_index)
        #     else:
        #         for idx in tqdm(video_index):
        #             run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=information), idx)
        #             resize_dataset(idx, dsize=args.output_res, info_dict=information)
    print('Done!')