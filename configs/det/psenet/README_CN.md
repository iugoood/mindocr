[English](README.md) | 中文

# PSENet

<!--- Guideline: use url linked to abstract in ArXiv instead of PDF for fast loading.  -->

> PSENet: [Shape Robust Text Detection With Progressive Scale Expansion Network](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)

## 概述

### PSENet

PSENet是一种基于语义分割的文本检测算法。它可以精确定位具有任意形状的文本实例，而大多数基于anchor类的算法不能用来检测任意形状的文本实例。此外，两个彼此靠近的文本可能会导致模型做出错误的预测。因此，为了解决上述问题，PSENet还提出了一种渐进式尺度扩展算法（Progressive Scale Expansion Algorithm, PSE）,利用该算法可以成功识别相邻的文本实例[[1](#参考文献)]。

<p align="center"><img alt="Figure 1. Overall PSENet architecture" src="https://github.com/VictorHe-1/mindocr_pse/assets/80800595/6ed1b691-52c4-4025-b256-a022aa5ef582" width="800"/></p>
<p align="center"><em>图 1. PSENet整体架构图</em></p>

PSENet的整体架构图如图1所示，包含以下阶段:

1. 使用Resnet作为骨干网络，从2，3，4，5阶段进行不同层级的特征提取；
2. 将提取到的特征放入FPN网络中，提取不同尺度的特征并拼接；
3. 将第2阶段的特征采用PSE算法生成最后的分割结果，并生成文本边界框。

### 配套版本

| mindspore  | ascend driver  |    firmware    | cann toolkit/kernel |
|:----------:|:--------------:|:--------------:|:-------------------:|
|   2.3.1    |    24.1.RC2    |  7.3.0.1.231   |    8.0.RC2.beta1    |

## 快速上手

### 安装

请参考MindOCR套件的[安装指南](https://github.com/mindspore-lab/mindocr#installation) 。

### 数据准备

#### ICDAR2015 数据集

请从[该网址](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载ICDAR2015数据集，然后参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示：

``` text
.
├── test
│   ├── images
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   └── test_det_gt.txt
└── train
    ├── images
    │   ├── img_1.jpg
    │   ├── img_2.jpg
    │   └── ....jpg
    └── train_det_gt.txt
```

#### SCUT-CTW1500 数据集

请从[该网址](https://github.com/Yuliang-Liu/Curve-Text-Detector)下载SCUT-CTW1500数据集，然后参考[数据转换](https://github.com/mindspore-lab/mindocr/blob/main/tools/dataset_converters/README_CN.md)对数据集标注进行格式转换。

完成数据准备工作后，数据的目录结构应该如下所示：

```txt
ctw1500
 ├── test_images
 │   ├── 1001.jpg
 │   ├── 1002.jpg
 │   ├── ...
 ├── train_images
 │   ├── 0001.jpg
 │   ├── 0002.jpg
 │   ├── ...
 ├── test_det_gt.txt
 ├── train_det_gt.txt
```

### 配置说明

在配置文件`configs/det/psenet/pse_r152_icdar15.yaml`中更新如下文件路径。其中`dataset_root`会分别和`data_dir`以及`label_file`拼接构成完整的数据集目录和标签文件路径。

```yaml
...
train:
  ckpt_save_dir: './tmp_det'
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: train/images                <--- 更新
    label_file: train/train_det_gt.txt    <--- 更新
...
eval:
  dataset_sink_mode: False
  dataset:
    type: DetDataset
    dataset_root: dir/to/dataset          <--- 更新
    data_dir: test/images                 <--- 更新
    label_file: test/test_det_gt.txt      <--- 更新
...
```

> 【可选】可以根据CPU核的数量设置`num_workers`参数的值。



PSENet由3个部分组成：`backbone`、`neck`和`head`。具体来说:

```yaml
model:
  type: det
  transform: null
  backbone:
    name: det_resnet152
    pretrained: True    # 是否使用ImageNet数据集上的预训练权重
  neck:
    name: PSEFPN         # PSENet的特征金字塔网络
    out_channels: 128
  head:
    name: PSEHead
    hidden_size: 256
    out_channels: 7     # kernels数量
```

### 训练
* 后处理

训练前，请确保在/mindocr/postprocess/pse目录下按照以下方式编译后处理代码：

``` shell
python3 setup.py build_ext --inplace
```

* 单卡训练

请确保yaml文件中的`distribute`参数为False。

``` shell
# train psenet on ic15 dataset
python tools/train.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

* 分布式训练

请确保yaml文件中的`distribute`参数为True。

```shell
# worker_num代表分布式总进程数量。
# local_worker_num代表当前节点进程数量。
# 进程数量即为训练使用的NPU的数量，单机多卡情况下worker_num和local_worker_num需保持一致。
msrun --worker_num=8 --local_worker_num=8 python tools/train.py --config configs/det/psenet/pse_r152_icdar15.yaml

# 经验证，绑核在大部分情况下有性能加速，请配置参数并运行
msrun --bind_core=True --worker_num=8 --local_worker_num=8 python tools/train.py --config configs/det/psenet/pse_r152_icdar15.yaml
```
**注意:** 有关 msrun 配置的更多信息，请参考[此处](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html).

训练结果（包括checkpoint、每个epoch的性能和曲线图）将被保存在yaml配置文件的`ckpt_save_dir`参数配置的路径下，默认为`./tmp_det`。

### 评估

评估环节，在yaml配置文件中将`ckpt_load_path`参数配置为checkpoint文件的路径，设置`distribute`为False，然后运行：

``` shell
python tools/eval.py --config configs/det/psenet/pse_r152_icdar15.yaml
```

### MindSpore Lite 推理

请参考[MindOCR 推理](../../../docs/zh/inference/inference_tutorial.md)教程，基于MindSpore Lite在Ascend 310上进行模型的推理，包括以下步骤：

- 模型导出

请先[下载](#2-实验结果)已导出的MindIR文件，或者参考[模型导出](../../../docs/zh/inference/convert_tutorial.md#1-模型导出)教程，使用以下命令将训练完成的ckpt导出为MindIR文件:

```shell
python tools/export.py --model_name_or_config psenet_resnet152 --data_shape 1472 2624 --local_ckpt_path /path/to/local_ckpt.ckpt
# or
python tools/export.py --model_name_or_config configs/det/psenet/pse_r152_icdar15.yaml --data_shape 1472 2624 --local_ckpt_path /path/to/local_ckpt.ckpt
```

其中，`data_shape`是导出MindIR时的模型输入Shape的height和width，下载链接中MindIR对应的shape值见[注释](#注释)。

- 环境搭建

请参考[环境安装](../../../docs/zh/inference/environment.md)教程，配置MindSpore Lite推理运行环境。

- 模型转换

请参考[模型转换](../../../docs/zh/inference/convert_tutorial.md#2-mindspore-lite-mindir-转换)教程，使用`converter_lite`工具对MindIR模型进行离线转换。

- 执行推理

在进行推理前，请确保PSENet的后处理部分已编译，参考[训练](#34-训练)的后处理部分。

假设在模型转换后得到output.mindir文件，在`deploy/py_infer`目录下使用以下命令进行推理：

```shell
python infer.py \
    --input_images_dir=/your_path_to/test_images \
    --det_model_path=your_path_to/output.mindir \
    --det_model_name_or_config=../../configs/det/psenet/pse_r152_icdar15.yaml \
    --res_save_dir=results_dir
```

## 性能表现

PSENet在ICDAR2015，SCUT-CTW1500数据集上训练。另外，我们在ImageNet数据集上进行了预训练，并提供预训练权重下载链接。所有训练结果如下：

### ICDAR2015

| **model name** | **backbone**  | **pretrained** | **cards** | **batch size** | **jit level** | **graph compile** | **ms/step** | **img/s** | **recall** | **precision** | **f-score** |               **recipe**               |                                             **weight**                                            |
|:--------------:|:-------------:| :------------: |:---------:|:--------------:| :-----------: |:-----------------:|:-----------:|:---------:|:----------:|:-------------:|:-----------:|:--------------------------------------:|:-------------------------------------------------------------------------------------------------:|
|     PSENet     |  ResNet-152   |    ImageNet    |     8     |       8        |      O2       |     225.02 s      |   355.19    |  180.19   |   78.91%   |    84.70%     |   81.70%    |    [yaml](pse_r152_icdar15.yaml)       | [ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ic15-6058a798-0d755205.mindir)   |
|     PSENet     |   ResNet-50   |    ImageNet    |     1     |       8        |      O2       |     185.16 s      |   280.21    |  228.40   |   76.55%   |    86.51%     |   81.23%    |      [yaml](pse_r50_icdar15.yaml)      | [ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet50_ic15-7e36cab9.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet50_ic15-7e36cab9-cfd2ee6c.mindir)    |
|     PSENet     |  MobileNetV3  |    ImageNet    |     8     |       8        |      O2       |     181.54 s      |   175.23    |  365.23   |   73.95%   |    67.78%     |   70.73%    |      [yaml](pse_mv3_icdar15.yaml)      |[ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_mobilenetv3_ic15-bf2c1907.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_mobilenetv3_ic15-bf2c1907-da7cfe09.mindir) |

### SCUT-CTW1500

| **model name** | **backbone**  | **pretrained** | **cards** | **batch size** | **jit level** | **graph compile** | **ms/step** | **img/s** | **recall** | **precision** | **f-score** |               **recipe**               |                                             **weight**                                            |
|:--------------:|:-------------:| :------------: |:---------:|:--------------:| :-----------: |:-----------------:|:-----------:|:---------:|:----------:|:-------------:|:-----------:|:--------------------------------------:|:-------------------------------------------------------------------------------------------------:|
|     PSENet     |  ResNet-152   |    ImageNet    |     8     |       8        |      O2       |     193.59 s      |   318.94    |  200.66   |   74.11%   |    73.45%     |   73.78%    |    [yaml](pse_r152_ctw1500.yaml)       | [ckpt](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ctw1500-58b1b1ff.ckpt) \| [mindir](https://download.mindspore.cn/toolkits/mindocr/psenet/psenet_resnet152_ctw1500-58b1b1ff-b95c7f85.mindir)  |

#### 注释：
- PSENet的训练时长受数据处理部分超参和不同运行环境的影响非常大。
- 在ICDAR15数据集上，以ResNet-152为backbone的MindIR导出时的输入Shape为`(1,3,1472,2624)` ，以ResNet-50或MobileNetV3为backbone的MindIR导出时的输入Shape为`(1,3,736,1312)`。
- 在SCUT-CTW1500数据集上，MindIR导出时的输入Shape为`(1,3,1024,1024)` 。

## 参考文献

<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] Wang, Wenhai, et al. "Shape robust text detection with progressive scale expansion network." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
