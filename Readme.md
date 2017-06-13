# Face Verification(Updating)
特征提取使用的是[Inception-ResNet](http://arxiv.org/abs/1602.07261.)模型，因为计算资源有限,直接用了[David Sandberg](https://github.com/davidsandberg)在MS-Celeb-1M数据集上的训练[结果](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit)，在这里作为分类器之前的特征提取部分。

## 代码结构
<pre>
code/
├── matlab
│   ├── distinguish.m
│   ├── get_labels.m
│   ├── Resize.m
│   └── readme.txt
├── models
|   └── inception_resnet_v1.py
├── classifier_face.py
├── demo_test.py
├── Face_Verification_batch.py
├── facenet.py
├── FaceVerification.py
├── lfw.py
├── sparse_mlp.py
└── train_model.py
</pre>

matlab文件夹里面的函数正如名字，用来Resize图片，划分训练集测试集以及写pairs文件,人脸对齐请参考[这里](https://github.com/kpzhang93/MTCNN_face_detection_alignment)。

## 实验效果
先试了下100人的情况，分类器用的SVM，1w多张图的训练时间还是很快的。

`python code/train_model.py`

    Number of classes: 100
    Number of images: 11416
    Loading feature extraction model
    Calculating features for images
    Calculating features Time 331.158
    Training classifier
    Training classifier Time 38.747
    Saved classifier model to file "./models/my_classifier.pkl"
人脸识别的准确度为93.3%。

`python code/classifier_svm.py`

    Number of classes: 100
    Number of images: 2862
    Loading feature extraction model
    Calculating features for images
    Calculating features Time 88.860
    Testing classifier
    Loaded classifier model from file "./models/my_classifier.pkl"
    classifier Time 9.956
    Accuracy: 0.933
这边是对12k个人脸pairs进行识别，判断他们是不是同一个人，整体的准确率是95.8%。和预想的一样，识别不是同一个人的准确度好于正确识别是同一个人的。

`python code/Face_Verification_batch.py`

    Loading feature extraction model
    Runnning forward pass on Validate images
    Calculating features Time 683.673
    Loaded classifier model from file "./models/my_classifier.pkl"
    Error Validated pairs have been logged
    Validate Time 80.237
    Pairs(total,same,diff): 12000 6000 6000
    Accuracy(total,same,diff): 0.958 0.918 0.997

当分类数达到10k+，数据样本有40w+时候，用SVM就不合适了，会计算量过大，这边用的是FNN对降维的特征进行分类。相比与100个类的时候，识别率有所下降，为86.5%，但是对人脸对的判别还好，是94.2%。

`python code/train_model.py`

    Number of classes: 10575
    Number of images: 407581
    Loading feature extraction model
    Calculating features for images
    Calculating features Time 11216.844
    Embeddings data saved
    Training classifier
    ...
    Accuracy at step 3500 for test set: 0.8756
    ...

`python code/classifier_face.py`

    Number of classes: 10575
    Number of images: 45071
    Loading feature extraction model
    Calculating features for images
    Calculating features Time 1160.829
    classifier Time 4.051
    Accuracy: 0.865

`python code/Face_Verification_batch.py`

    Loading feature extraction model
    Runnning forward pass on Validate images
    Calculating features Time 857.511
    Loaded classifier model from file ...
    Error Validated pairs have been logged.
    Validate Time 3.902
    Pairs(total,same,diff): 17509 7511 9998
    Accuracy(total,same,diff): 0.942 0.864 1.000








