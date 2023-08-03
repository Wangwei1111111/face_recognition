# face_recognition
face recognition based on FaceNet
<br>
<br>
## 使用开源facenet模型

（https://github.com/davidsandberg/facenet）

### facenet原理：    
* 1. 将图像通过深度卷积神经网络映射到128维的特征空间（欧式空间）中，得到对应的128维特征向量；
     
* 2. 对特征向量进行L2正则化，筛选出有效特征；
     
* 3. 使用正则化后的特征向量，计算***Triplets Loss***。  

### 训练：  

train_tripletloss.py

### 验证数据集：  

使用[LFW人脸比对数据集](http://vis-www.cs.umass.edu/lfw/index.html#download)验证，下载173MB那个  

* 1. 使用align_dataset_mtcnn.py裁剪、对齐、标记人脸，裁剪成160尺寸
* 2. 使用validate_on_lfw.py验证，输出精度

### 目录：  

/src：算法实现  
    
/contributed：现成的一些功能  

### /src/  

compare.py：输入模型和多张图片，进行裁剪、识别、计算特征向量。输出不同人脸之间的欧氏距离。  


### /contributed/  

predict.py：使用mtcnn进行人脸检测并对齐与裁剪；对裁剪的人脸使用facenet进行embedding；执行predict.py进行人脸识别（需要训练好的svm模型）  
    
cluster.py：使用mtcnn进行人脸检测并对齐与裁剪；对裁剪的人脸使用facenet进行embedding；对embedding的特征向量使用欧式距离进行聚类  
    
real_time_face_recognition.py：开启摄像头，捕获帧，调用face.py识别人脸、计算特征向量，找到匹配的标签，在视频中显示（需要训练好的svm模型）  
<br>
<br>
## Myface 人脸识别验证  

### 功能：  

开启摄像头，实时识别人脸，与数据库中的人脸信息进行配对，输出人员信息。  

### 数据库：  

Oracle DWD ZT_RKK_RLSB（pid，facecoding）  

### add_person.py：  

* 输入参数图片路径；  
* 裁剪、识别人脸、加载模型；  
* 获得特征向量（128维）；  
* 输入对应人员信息，将特征向量转为二进制（BLOB）存入数据库。  

### person.py：  

* 传入参数为视频获取的帧图像；  
* 裁剪、识别人脸、加载模型、获得特征向量（128维）embedding；  
* 从数据库中取出所有特征向量face_coding（由二进制转回向量），以及对应人员pid；  
* 以数据库中的所有特征向量为训练集训练KNN算法，比对embedding和face_coding；  
* 找到距离最近（k=1）的一个，判断欧式距离是否小于设定阈值（0.8）；  
* 返回person类。  

### camera_face_recognition.py：  
* 开启摄像头，死循环捕获帧图像；  
* 将帧图像传递给person.py，接收返回的person类；  
* 将person信息实时显示在视频中（add_overlays覆盖）。  

### 配置环境：  

Python3.6，tensorflow-gpu1.7，cuda9，cudnn7  

其他在requirements.txt中，可以使用`pip install -r requirements.txt`安装  

注：使用官方基于VGGFace2的预训练模型[20180402-114759](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view)，加载模型加载.pb文件  
<br>
<br>
<br>
<br>
智慧岛  平潭  2023.08  
小鸡蛋
