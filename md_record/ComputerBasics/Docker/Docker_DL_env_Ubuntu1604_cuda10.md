# Docker利用cuda镜像，基于Ubuntu16.04搭建深度学习环境

*   Ubuntu16.04

*   Cuda10.0

*   cudnn7

1.  拉取镜像

```
docker pull nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
```

>   <https://hub.docker.com/r/nvidia/cuda/tags/?page=1&name=10.0&ordering=-last_updated>

2\.   生成容器

    
    docker run -it --runtime=nvidia -v /宿主机绝对路径目录:/容器内目录 --name 容器名 镜像名 /bin/bash
    
    # 我用的比较简单
    
    docker run -it \<image\_id> bash

3\.   在容器内安装深度学习环境

```
#!/bin/bash
set -e
 
apt-get update
#安装vim
apt-get -y install vim
 
#可选，解决vim中文乱码
#vim /etc/vim/vimrc
#set fileencodings=utf-8,gbk,utf-16le,cp1252,iso-8859-15,ucs-bom
#set termencoding=utf-8
#set encoding=utf-8
#
#安装sqlite3
apt-get install libsqlite3-dev

# 可选
# apt-get -y install gcc
 
#解决ssl No module named _ssl
apt-get install libssl-dev -y

# 安装make
apt-get install make
 
# 安装zlib*
apt-get -y install zlib*
 
#安装wget
apt-get -y install wget
 
#下载python
wget https://www.python.org/ftp/python/3.7.10/Python-3.7.10.tgz

# 安装依赖
apt-get install libffi-dev libsqlite3-dev libbz2-dev liblzma-dev
apt-get install -y  zlib1g-dev libbz2-dev libssl-dev libncurses5-dev libsqlite3-dev
apt-get install -y make build-essential python-dev libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl libice-doc liblzma-doc ncurses-doc readline-doc libsm-doc sqlite3-doc libxcb-doc libxext-doc libxt-doc python-cryptography-doc python-cryptography-vectors python-enum34-doc python-openssl-doc  python-setuptools tcl-doc tcl8.6-doc tk-doc tk8.6-doc


#解压安装Python
tar -xvzf Python-3.7.10.tgz
cd Python-3.7.10
./configure --prefix=/opt/python3.7 --with-ssl
make && make install

ln -s /usr/local/bin/python3 /usr/bin/python3
ln -s /usr/local/bin/pip3 /usr/bin/pip

#升级pip
pip install --upgrade pip setuptools

#解决opencv 报错安装依赖
apt-get install -y build-essential libgtk2.0-dev libavcodec-dev libavformat-dev libjpeg.dev libtiff4.dev libswscale-dev libjasper-dev
sudo apt-get install -y cmake git pkg-config   
sudo apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev 

apt-get -y install libgl1-mesa-glx libglib2.0-dev 


 
#删除安装文件
#rm -R Python*

 
pip3 install -r requirements.txt -i https://pypi.douban.com/simple
 
 
#mysql
#cd /home
#wget https://dev.mysql.com/get/mysql-apt-config_0.8.15-1_all.deb
 
 
#中文编码问题
export  LANG=C.UTF-8
 
#解决cv2 问题
#apt-get -y install libgl1-mesa-glx
#python -V


```

1.  Cudnn 7.4.1

    安装cudnn

    ```

    sudo cp cuda/include/cudnn.h /usr/local/cuda-10.0/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64/

    # 更改文件权限
    sudo chmod a+r /usr/local/cuda-10.0/include/cudnn.h 
    sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*

    # 检查cudnn是否安装成功
    cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A2
    ```

2.  报错

    **apt update** 时会报错。

        W: GPG error: <https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64> InRelease: The following signatures couldn’t be verified because the public key is not available: NO\_PUBKEY A4B469963BF863CC
        E: The repository ‘<https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64> InRelease’ is not signed.
        N: Updating from such a repository can’t be done securely, and is therefore disabled by default.
        N: See apt-secure(8) manpage for repository creation and user configuration details.

    解决方法：

        apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC

    引用tensorflow，报错1：

        ImportError: libcuda.so.1: cannot open shared object file: No such file or directory

    解决

        # 查找libcuda.so.1
        find /usr -name libcuda.so.1
        
        # 在修改～/.bashrc文件，在其中增加一条
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your install path

    报错2：

        TypeError: Descriptors cannot not be created directly.
        If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
        If you cannot immediately regenerate your protos, some other possible workarounds are:
         1. Downgrade the protobuf package to 3.20.x or lower.
         2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

    解决：降低protobuf版本

    ```
    pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple protobuf==3.20
    
    ```

