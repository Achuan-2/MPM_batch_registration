"""批量进行配准
运行方式：
    1. 在命令行中运行：
        1. cd 本文件夹
        2. conda activate suite2p
        3.python batch_registration.py
    2. 运行 batch_registration.sh
    3. 在pycharm/VSCode中，选择conda环境后，直接运行本文件
运行中：弹出弹窗，选择包含tif文件的文件夹，进行配准
运行结果：
    如果选择的文件夹下，有多个子文件夹，那么会对每个子文件夹进行配准
    如果选择的文件夹下，没有子文件夹，那么会对该文件夹下的所有tif文件进行配准
    配准后的文件会保存在选择的文件夹下的同级目录下，文件夹名为：选择的文件夹名_registration
"""
import os
import numpy as np
from default_ops import reg_default_ops
from registration import register
from tifffile import imread, TiffFile, TiffWriter
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def main():
    root = tk.Tk()
    root.withdraw()
    parent_folder_path = filedialog.askdirectory()  # Ask the user to select a folder
    if not parent_folder_path:
        print("\033[91m" + "No folder selected. Program is terminated. " + "\033[0m")
        return
    print("Choose Path:", parent_folder_path)

    # Get all subfolders
    subfolders = [
        f.path
        for f in os.scandir(parent_folder_path)
        if f.is_dir() and not f.name.endswith("_registration")
    ]
    # If no subfolders, consider the parent folder for processing
    if not subfolders:
        folder_path = parent_folder_path
        output_folder = os.path.join(
            os.path.dirname(parent_folder_path),
            os.path.basename(parent_folder_path) + "_registration",
        )
        os.makedirs(output_folder, exist_ok=True)

        # Reg all tif files
        reg_folder_tif(folder_path, output_folder)
    else:
        for folder_path in subfolders:
            # Create the output folder
            output_folder = os.path.join(
                os.path.dirname(parent_folder_path),
                os.path.basename(parent_folder_path) + "_registration",
                os.path.basename(folder_path),
            )
            os.makedirs(output_folder, exist_ok=True)

            # Reg all tif files
            reg_folder_tif(folder_path, output_folder)


def reg_folder_tif(folder_path, output_folder):
    # Get all tif files
    tif_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".tif") and os.path.isfile(os.path.join(folder_path, f))
    ]
    print(f"----------REGISTRATION Start for {folder_path}")
    progress_bar = tqdm(tif_files)
    for filename in progress_bar:
        progress_bar.set_description(f"Processing {filename}")
        if filename.endswith(".tif"):
            input_path = os.path.join(folder_path, filename)
            output_path = os.path.join(
                output_folder, filename.replace(".tif", "_reg.tif")
            )

            reg_single_tif(input_path, output_path)


def reg_single_tif(input_path, output_path):
    ops = reg_default_ops()

    input_image = imread(input_path)

    registered_image = reg(input_image, ops)

    with TiffWriter(output_path) as tif:
        tif.write(np.floor(registered_image).astype(np.int16))


def reg(raw_image, ops):
    n_frames, Ly, Lx = raw_image.shape
    ops["batch_size"] = n_frames
    Midrefimage = register.compute_reference(raw_image, ops)
    (
        maskMul,
        maskOffset,
        cfRefImg,
        maskMulNR,
        maskOffsetNR,
        cfRefImgNR,
        blocks,
    ) = register.compute_reference_masks(Midrefimage, ops)
    refAndMasks = [
        maskMul,
        maskOffset,
        cfRefImg,
        maskMulNR,
        maskOffsetNR,
        cfRefImgNR,
        blocks,
    ]
    (
        reg_image,
        ymax,
        xmax,
        cmax,
        ymax1,
        xmax1,
        cmax1,
        nonsense,
    ) = register.register_frames(
        refAndMasks, raw_image, rmin=-np.inf, rmax=np.inf, bidiphase=0, ops=ops, nZ=1
    )

    return np.floor(reg_image).astype(np.int16)


if __name__ == "__main__":
    main()
