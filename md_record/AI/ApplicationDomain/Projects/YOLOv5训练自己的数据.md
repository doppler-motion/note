# YOLOv5训练自己的数据

## 环境配置

*   编程环境

> pytorch： 1.13
>
> python：3.8
>
> yolov5

*   yolov5下载

> 下载源码：<https://github.com/ultralytics/yolov5>
>
> 进入 yolov5 文件夹，安装依赖。

### 准备数据

*   可在 **YOLOv5** 或者 **YOLOv5/data** 目录下建立自己的数据目录 **mydata**

*   在 **mydata** 下，新建 **images** 和 **labels** （coco格式目录划分，还有coco128格式）

    > images : 存放图片
    >
    > labels： 存放图片标注后的内容，YOLO格式的

### 数据格式转换

*   voc格式的标注格式（xml）需要转换成yolo格式（txt）的

### 数据集划分

*   训练集、测试集、验证集划分

    > 在 **mydata** 下，新建py文件  traindata\_split.py， 名字任取

    ```python
    import os
    import random

    total = 1.0
    train_percent = 0.9
    test_percent = 0.1

    datapath = "images"  # 图片目录
    split_dataset = "dataSet/Main"  # 存放图片文件名的目录
    split_data_path = "dataSet/Path"  # 存放图片文件路径的目录

    imgs_list = os.listdir(datapath) # 文件列表
    total_nums = len(imgs_list)  # 文件总数
    list_index = range(total_nums)

    tv = int(total_nums * total)
    tr = int(tv * train_percent)
    train_index = random.sample(list_index, tv)
    train = random.sample(train_index, tr)

    if not os.path.exists(split_dataset):
        os.mkdir(split_dataset)
    if not os.path.exists(split_data_path):
        os.mkdir(split_data_path)

    file_trainval = open(split_dataset + "/total.txt", "w")
    file_train = open(split_dataset + "/train.txt", "w")
    file_test = open(split_dataset + "/test.txt", "w")
    file_val = open(split_dataset + "/val.txt", "w")

    file_trainval_path = open(split_data_path + "/total.txt", "w")
    file_train_path = open(split_data_path + "/train.txt", "w")
    file_test_path = open(split_data_path + "/test.txt", "w")
    file_val_path = open(split_data_path + "/val.txt", "w")

    wd = os.getcwd()
    for i in list_index:
        name = imgs_list[i][:-4] + "\n"
        img_path = os.path.join(wd, datapath, imgs_list[i]).replace("\\", "/") + "\n"

        if i in train_index:
            file_trainval.write(name)
            file_trainval_path.write(img_path)
            if i in train:
                file_train.write(name)
                file_train_path.write(img_path)
            else:
                file_val.write(name)
                file_val_path.write(img_path)
        else:
            file_test.write(name)
            file_test_path.write(img_path)

    file_trainval.close()
    file_train.close()
    file_val.close()
    file_test.close()

    file_trainval_path.close()
    file_train_path.close()
    file_val_path.close()
    file_test_path.close()

    ```

    > 会建立 **dataSet/Main** 和 **dataSet/Path** 两个目录，且分别存有 total.txt, train.txt, val.txt, test.txt 四个文件，
    >
    > **dataSet/Main** 里面的文件保存图片的名字（无后缀.jpg）
    >
    > **dataSet/Path** 里面的文件保存图片的绝对路径

    > 此脚本没有设置测试集，故两个目录下的 test.txt 均为空

*   配置文件

    > 在 YOLOv5 的 **data** 目录下， 新建  mydata.yaml 文件（文件名任取），填入以下内容：
    >
    > *   训练集以及验证集（**train.txt** 和 **val.txt** ）的绝对路径（**可以改为相对路径**）（\*\*这两个txt文件是保存图片文件路径的txt文件）
    > *   目标的类别数目
    > *   类别名称

    代码示例

    ```yaml
    train: D:/Yolov5/yolov5/VOCData/dataSet_path/train.txt
    val: D:/Yolov5/yolov5/VOCData/dataSet_path/val.txt

    # number of classes
    nc: 2

    # class names
    names: ["light", "post"]
    ```

*   模型配置文件

    根据使用的模型，修改相应模型的配置文件。比如，使用s，则在 YOLOv5的 **model** 目录下，修改 yolov5s.yaml

    > 将 nc 改为 实际的类别数

### 模型训练

#### 训练指令

    python train.py --weights weights/yolov5s.pt  --cfg models/yolov5s.yaml  --data data/mydata.yaml --epoch 200 --batch-size 4 --img 640  --device 0

>

### 模型推理

    python detect.py --weights runs/train/exp21/weights/best.pt --source test_/e0ec7c95ae18d5353faa3f521d08d85c_001732.jpg

> 其中， e0ec7c95ae18d5353faa3f521d08d85c\_001732.jpg为测试图片,
>
> runs/train/exp21/weights/best.pt 刚跑出的模型

