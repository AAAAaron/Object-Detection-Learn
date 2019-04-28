# 一些项目的运行情况（场景文字检测，中英文文字识别）

[toc]

## [AdvancedEAST](https://github.com/huoyijie/AdvancedEAST)

屏幕文本检测方法，基于EAST的改进方法，可以检测倾斜文本，是通过检测文本头和文本尾来保证的，效果较为相比原来的那种文字联通方法来看，更准确一点。
但是在实际使用的时候发现，有时候某个文本会只检测出头或者只检测出尾巴，这样会导致在后面合成的时候就丢失了，这是不太好的地方，虽然看到Blok里面有人说可以改进这个地方
但是一旦改进，就是把各种倾斜弯曲文本又变成了直线，某些情况下不太好。
这个对于标准场景或者图片的检测效果确实很好，但是自然环境下的文字检测就打折扣了

## [ASTER: Attentional Scene Text Recognizer with Flexible Rectification](https://github.com/bgshih/aster)

图片的文字识别方法，较新，但是目前还没有炮通

## [chinese_ocr](https://github.com/YCG09/chinese_ocr)

基于Tensorflow和Keras实现端到端的不定长中文字符检测和识别
在p2的环境下可以运行，目标应该是倾斜文本也可以提取
但是无论是图片区域提取还是里面的识别，感觉都不能很好的实现效果
识别也并不是很准。
可能是ctpn对倾斜文本或者弯曲文本的检测效果并不好
目前来说可能是模型里没有训练英文，导致给英文出来的都是乱的汉字

## [puqiu/chinses-ocr](https://github.com/puqiu/chinses-ocr)

本文基于tensorflow、keras实现对自然场景的文字检测及端到端的OCR中文文字识别
还没试

## [CHINESE-OCR](https://github.com/xiaofengShi/CHINESE-OCR)

一共分为3个网络 ：

1. 文本方向检测网络-Classify(vgg16)
2. 文本区域检测网络-CTPN(CNN+RNN)
3. EndToEnd文本识别网络-CRNN(CNN+GRU/LSTM+CTC)

开始是可以运行的，不知道环境出了什么变化，后来运行的时候就一直报错：

``` python
ValueError: Shape must be rank 1 but is rank 0 for 'batch_normalization_1/cond/Reshape_4' (op: 'Reshape') with input shapes: [1,4,1,1], [].
```

## [crnn.pytorch](https://github.com/meijieru/crnn.pytorch)

 (CRNN) in pytorch的一个实现，其中效果对于自然场景下，比较清楚的照片识别是比较好的，特别是对于英文的识别，效果会更好一些

## [crnn](https://github.com/bgshih/crnn)

这个是另一个人CRNN的torch实现，效果还没下载，还需要测试，但是看readme好像是和上面那个是一样的

## [FCOS](https://github.com/tianzhi0549/FCOS)

Fully Convolutional One-Stage Object Detection
一段式的全卷积神经网络检测，是一种新的无框方法，检测效果未知，只是安装完成了

## [FOTS](https://github.com/xieyufei1993/FOTS)

This is a pytorch re-implementation of FOTS: Fast Oriented Text Spotting with a Unified Network. The features are summarized blow:
他说只完成了检测部分，下载还没有跑通

## [keras-ctpn](https://github.com/yizt/keras-ctpn)

一个较好的keras实现的ctpn，很详细的讨论了很多关于ctpn的问题，可以完成对于倾斜文本的检测，![但是有效的框会变少](./result/kerasCtpn.jpg)，可能跟他的阈值有关系


4.1 [ICDAR2015](#ICDAR2015)
4.1.1 [带侧边细化](#带侧边细化)
4.1.2 [不带带侧边细化](#不带侧边细化)
4.1.3 [做数据增广-水平翻转](#做数据增广-水平翻转)
4.2 [ICDAR2017](#ICDAR2017)
4.3 [其它数据集](#其它数据集)

## [text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)

这个库可以说是非常经典了，在很多地方的库里关于ctpn的部分都是引用自这个库
，他是参考了一个旧的caffe版本，我觉得使用下来，这个读起来是最明确清楚的代码，使用tensorflow语法也很简洁，效果也比较稳定，只是这个库不能完成倾斜文本的检测，使用的也是原论文的文本联通方法.

## [chineseocr](https://github.com/chineseocr/chineseocr)

这是另一个实现的版本，使用yolo3 与crnn 实现中文自然场景文字检测及识别，算是圆了我之前对于yolo在文字场景识别上的猜想，目前还没有完全测试，稍后测试

## [Mask_RCNN](https://github.com/matterport/Mask_RCNN)

基于  Feature Pyramid Network (FPN) 和 a ResNet101 backbone 在 Python 3, Keras, and TensorFlow的 Mask R-CNN，效果看起来比较好，代码还没看完，主要用于目标检测方向

## 还有几个EAST的方法没有看