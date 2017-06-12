# Face Verification(Updating)
特征提取使用的是[Inception-ResNet](http://arxiv.org/abs/1602.07261.)模型并用到了[David Sandberg](https://github.com/davidsandberg)在MS-Celeb-1M数据集上的训练[结果](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit)

先试了下100人的情况，分类器用的SVM

`python code/train_model.py`

    Number of classes: 100
    Number of images: 11416
    Loading feature extraction model
    Calculating features for images
    Calculating features Time 331.158
    Training classifier
    Training classifier Time 38.747
    Saved classifier model to file "./models/my_classifier.pkl"

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

`python code/Face_Verification_batch.py`

    Loading feature extraction model
    Runnning forward pass on Validate images
    Calculating features Time 683.673
    Loaded classifier model from file "./models/my_classifier.pkl"
    Error Validated pairs have been logged
    Validate Time 80.237
    Pairs(total,same,diff): 12000 6000 6000
    Accuracy(total,same,diff): 0.958 0.918 0.997

10k个人的时候，用SVM计算量过大，这边用的是FNN对降维的特征进行分类

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




