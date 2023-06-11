# Docker利用cuda镜像，基于Ubuntu16.04搭建深度学习环境

* Ubuntu16.04

* Cuda10.0

* cudnn7

  

1. 拉取镜像

```
docker pull nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
```



> https://hub.docker.com/r/nvidia/cuda/tags/?page=1&name=10.0&ordering=-last_updated

2. 生成容器

```
docker run -it --runtime=nvidia -v /宿主机绝对路径目录:/容器内目录 --name 容器名 镜像名 /bin/bash


# 我用的比较简单
docker run -it <image_id> bash
```

3. 在容器内安装深度学习环境

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
apt-get install libffi-dev
apt-get install libsqlite3-dev
apt-get install libbz2-dev
apt-get install liblzma-dev

#解压安装Python
tar -xvzf Python-3.7.10.tgz
cd Python-3.7.10
./configure --prefix=/opt/python3.7 --with-ssl
make && make install

ln -sf /usr/local/bin/python3 /usr/bin/python
 
#解决opencv 报错
apt-get -y install libgl1-mesa-glx
apt-get install -y libglib2.0-dev 
 
 
#删除安装文件
#rm -R Python*
 
#升级pip
python3 -m pip install --upgrade pip
 
pip3 install -r requirements.txt -i https://pypi.douban.com/simple
 
 
#mysql
#cd /home
#wget https://dev.mysql.com/get/mysql-apt-config_0.8.15-1_all.deb
 
 
#中文编码问题
export  LANG=C.UTF-8
 
#解决cv2 问题
#apt-get -y install libgl1-mesa-glx
#python -V
 
#升级pip
#python -m pip install --upgrade pip

```



4. Cudnn 7.4.1



报错

1. **apt update** 时会报错。

```
W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 InRelease: The following signatures couldn’t be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
E: The repository ‘https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 InRelease’ is not signed.
N: Updating from such a repository can’t be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.
```

解决方法：

```
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC
```

