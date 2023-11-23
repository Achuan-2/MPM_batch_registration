## 代码功能

批量进行配准，配准的代码来自[suite2p](https://github.com/MouseLand/suite2p)，只是将其封装成了一个函数，方便批量进行配准

## 运行代码

运行条件：可以直接复制本文件夹到任意文件夹下，只需要电脑配有suite2p环境即可

运行方式
1. 在命令行中运行：
    1. 进入本文件夹，在文件夹路径输入 `cmd` 打开终端
    2. 输入 `conda activate suite2p`，激活 conda 环境
    3. 运动代码 `python batch_registration.py`
2. 双击 `batch_registration.sh`
3. 在pycharm/VSCode中，选择 conda 环境后，直接运行本文件

## 运行成功

弹出弹窗，选择包含tif文件的文件夹，进行配准

## 运行结果

1. 如果选择的文件夹下，有多个子文件夹，那么会对每个子文件夹进行配准
2. 如果选择的文件夹下，没有子文件夹，那么会对该文件夹下的所有tif文件进行配准
3. 配准后的文件会保存在选择的文件夹下的同级目录下，文件夹名为：选择的文件夹名_registration

## 修改配准参数

在本文件夹下，打开 default_ops.py，修改参数


