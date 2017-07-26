本文主要介绍如何通过AI（人工智能）的方式玩Flappy Bird游戏，分为以下四个部分内容：

1、Flappy Bird 游戏展示
2、模型：卷积神经网络
3、算法：Deep Q Network
4、代码：TensorFlow实现

# 一、Flappy Bird 游戏展示
<img src="./images/flappy_bird_demp.gif" width="250">

7 mins version: [DQN for flappy bird](https://www.youtube.com/watch?v=THhUXIhjkCM)


# 二、模型：卷积神经网络

神经网络算法是由众多的神经元可调的连接权值连接而成，具有大规模并行处理、分布式信息存储、良好的自组织自学习能力等特点。人工神经元与生物神经元结构类似，其结构对比如下图所示。
人工神经元的输入（x1,x2...xm）类似于生物神经元的树突，输入经过不同的权值（wk1, wk2, ....wkn），加上偏置，经过激活函数得到输出，最后将输出传输到下一层神经元进行处理。



激活函数为整个网络引入了非线性特性，这也是神经网络相比于回归等算法拟合能力更强的原因。常用的激活函数包括sigmoid、tanh等，它们的函数表达式如下：




这里可以看出，sigmoid函数的值域是（0,1），tanh函数的值域是（-1,1）。

** 卷积神经网络**起源于动物的视觉系统，主要包含的技术是：

局部感知域（稀疏连接）；
参数共享；
多卷积核；
池化。
1. 局部感知域（稀疏连接）
全连接网络的问题在于：

需要训练的参数过多，容器导致结果不收敛（梯度消失），且训练难度极大；
实际上对于某个局部的神经元来讲，它更加敏感的是小范围内的输入，换句话说，对于较远的输入，其相关性很低，权值也就非常小。
人类的视觉系统决定了人在观察外界的时候，总是从局部到全局。

比如，我们看到一个美女，可能最先观察到的是美女身上的某些部位（自己体会）。
因此，卷积神经网络与人类的视觉类似，采用局部感知，低层的神经元只负责感知局部的信息，在向后传输的过程中，高层的神经元将局部信息综合起来得到全局信息。



从上图中可以看出，采用局部连接之后，可以大大的降低训练参数的量级。

2. 参数共享
虽然通过局部感知降低了训练参数的量级，但整个网络需要训练的参数依然很多。

参数共享就是将多个具有相同统计特征的参数设置为相同，其依据是图像中一部分的统计特征与其它部分是一样的。其实现是通过对图像进行卷积（卷积神经网络命名的来源）。

可以理解为，比如从一张图像中的某个局部（卷积核大小）提取了某种特征，然后以这种特征为探测器，应用到整个图像中，对整个图像顺序进行卷积，得到不同的特征。


每个卷积都是一种特征提取方式，就像一个筛子，将图像中符合条件（激活值越大越符合条件）的部分筛选出来，通过这种卷积就进一步降低训练参数的量级。

3. 多卷积核
如上，每个卷积都是一种特征提取方式，那么对于整幅图像来讲，单个卷积核提取的特征肯定是不够的，那么对同一幅图像使用多种卷积核进行特征提取，就能得到多幅特征图（feature map）。



多幅特征图可以看成是同一张图像的不同通道，这个概念在后面代码实现的时候用得上。

4. 池化
得到特征图之后，可以使用提取到的特征去训练分类器，但依然会面临特征维度过多，难以计算，并且可能过拟合的问题。从图像识别的角度来讲，图像可能存在偏移、旋转等，但图像的主体却相同的情况。也就是不同的特征向量可能对应着相同的结果，那么池化就是解决这个问题的。



池化就是将池化核范围内（比如2*2范围）的训练参数采用平均值（平均值池化）或最大值（最大值池化）来进行替代。

终于到了展示模型的时候，下面这幅图是笔者手画的（用电脑画太费时，将就看吧），这幅图展示了本文中用于训练游戏所用的卷积神经网络模型。





a. 初始输入四幅图像80×80×4（4代表输入通道，初始时四幅图像是完全一致的），经过卷积核8×8×4×32（输入通道4，输出通道32），步距为4（每步卷积走4个像素点），得到32幅特征图（feature map），大小为20×20；隐藏神经元个数是20×20×32 = 12800个

有必要把tensoflow的卷积公式拿出来说明一下： tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')
W是权重矩阵，其实对应的就是上面的卷集核, strides定义步长，[1,1,1,1]表示步长是1，卷集核从左到右逐行扫描，如果是[1,4,4,1]表示卷集核取一个卷集核大小，后跳4行，再取。padding表示边界处理方式，一般取SAME，表示卷积核输出和输入保持同样尺寸。

b. 将20×20的图像进行池化，池化核为2×2，得到图像大小为10×10；特征图数量不变，池化后神经元个数：10×10×32=3200个

c. 再次卷积，卷积核为4×4×32×64，步距为2，得到图像5×5×64；

d. 再次卷积，卷积核为3×3×64*64，步距为2，得到图像5×5×64，虽然与上一步得到的图像规模一致，但再次卷积之后的图像信息更为抽象，也更接近全局信息；

e. Reshape，即将多维特征图转换为特征向量，得到1600维的特征向量；

f. 经过全连接1600×512，得到512维特征向量；

g. 再次全连接512×2，得到最终的2维向量[0,1]和[1,0]，分别代表游戏屏幕上的是否点击事件。

可以看出，该模型实现了端到端的学习，输入的是游戏屏幕的截图信息（代码中经过opencv处理），输出的是游戏的动作，即是否点击屏幕。深度学习的强大在于其数据拟合能力，不需要传统机器学习中复杂的特征提取过程，而是依靠模型发现数据内部的关系。
