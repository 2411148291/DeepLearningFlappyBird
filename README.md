本文主要介绍如何通过AI（人工智能）的方式玩Flappy Bird游戏，分为以下四个部分内容：

1、Flappy Bird 游戏展示<br>
2、模型：卷积神经网络<br>
3、算法：Deep Q Network<br>
4、代码：TensorFlow实现<br>

# 一、Flappy Bird 游戏展示
<img src="./images/flappy_bird_demp.gif" width="250">

7 mins version: [DQN for flappy bird](https://www.youtube.com/watch?v=THhUXIhjkCM)


## 二、模型：卷积神经网络

神经网络算法是由众多的神经元可调的连接权值连接而成，具有大规模并行处理、分布式信息存储、良好的自组织自学习能力等特点。人工神经元与生物神经元结构类似，其结构对比如下图所示。<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/bi-neuron.png)
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/neurons.jpg)<br>
人工神经元的输入（x1,x2...xm）类似于生物神经元的树突，输入经过不同的权值（wk1, wk2, ....wkn），加上偏置，经过激活函数得到输出，最后将输出传输到下一层神经元进行处理。<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/f1.png)
<br>

激活函数为整个网络引入了非线性特性，这也是神经网络相比于回归等算法拟合能力更强的原因。常用的激活函数包括sigmoid、tanh等，它们的函数表达式如下：
<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/f2.png)<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/f3.png)
<br>
这里可以看出，sigmoid函数的值域是（0,1），tanh函数的值域是（-1,1）。

卷积神经网络 起源于动物的视觉系统，主要包含的技术是：

* 局部感知域（稀疏连接）；
* 参数共享；
* 多卷积核；
* 池化。
### 1. 局部感知域（稀疏连接）
全连接网络的问题在于：

需要训练的参数过多，容器导致结果不收敛（梯度消失），且训练难度极大；
实际上对于某个局部的神经元来讲，它更加敏感的是小范围内的输入，换句话说，对于较远的输入，其相关性很低，权值也就非常小。
人类的视觉系统决定了人在观察外界的时候，总是从局部到全局。

>比如，我们看到一个美女，可能最先观察到的是美女身上的某些部位（自己体会）。<br>

因此，卷积神经网络与人类的视觉类似，采用局部感知，低层的神经元只负责感知局部的信息，在向后传输的过程中，高层的神经元将局部信息综合起来得到全局信息。
<br><br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/fc1.png)
<br><br>
从上图中可以看出，采用局部连接之后，可以大大的降低训练参数的量级。

### 2. 参数共享
虽然通过局部感知降低了训练参数的量级，但整个网络需要训练的参数依然很多。

参数共享就是将多个具有相同统计特征的参数设置为相同，其依据是图像中一部分的统计特征与其它部分是一样的。其实现是通过对图像进行卷积（卷积神经网络命名的来源）。

可以理解为，比如从一张图像中的某个局部（卷积核大小）提取了某种特征，然后以这种特征为探测器，应用到整个图像中，对整个图像顺序进行卷积，得到不同的特征。
<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/share.gif)
<br>
每个卷积都是一种特征提取方式，就像一个筛子，将图像中符合条件（激活值越大越符合条件）的部分筛选出来，通过这种卷积就进一步降低训练参数的量级。

### 3. 多卷积核
如上，每个卷积都是一种特征提取方式，那么对于整幅图像来讲，单个卷积核提取的特征肯定是不够的，那么对同一幅图像使用多种卷积核进行特征提取，就能得到多幅特征图（feature map）。
<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/fc2.png)
<br>
多幅特征图可以看成是同一张图像的不同通道，这个概念在后面代码实现的时候用得上。

### 4. 池化
得到特征图之后，可以使用提取到的特征去训练分类器，但依然会面临特征维度过多，难以计算，并且可能过拟合的问题。从图像识别的角度来讲，图像可能存在偏移、旋转等，但图像的主体却相同的情况。也就是不同的特征向量可能对应着相同的结果，那么池化就是解决这个问题的。
<br><br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/pool.gif)
<br><br>
池化就是将池化核范围内（比如2×2范围）的训练参数采用平均值（平均值池化）或最大值（最大值池化）来进行替代。
终于到了展示模型的时候，下面这幅图是笔者手画的（用电脑画太费时，将就看吧），这幅图展示了本文中用于训练游戏所用的卷积神经网络模型。
<br><br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/blackboard.png)<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/illus.png)
<br><br>
a. 初始输入四幅图像80×80×4（4代表输入通道，初始时四幅图像是完全一致的），经过卷积核8×8×4×32（输入通道4，输出通道32），步距为4（每步卷积走4个像素点），得到32幅特征图（feature map），大小为20×20；隐藏神经元个数是20×20×32 = 12800个

有必要把tensoflow的卷积公式拿出来说明一下： tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')
W是权重矩阵，其实对应的就是上面的卷集核, strides定义步长，[1,1,1,1]表示步长是1，卷集核从左到右逐行扫描，如果是[1,4,4,1]表示卷集核取一个卷集核大小，后跳4行，再取。padding表示边界处理方式，一般取SAME，表示卷积核输出和输入保持同样尺寸。

b. 将20×20的图像进行池化，池化核为2×2，得到图像大小为10×10；特征图数量不变，池化后神经元个数：10×10×32=3200个

c. 再次卷积，卷积核为4×4×32×64，步距为2，得到图像5×5×64；神经元个数：5×5×64=1600个

d. 再次卷积，卷积核为3×3×64*64，步距为2，得到图像5×5×64，虽然与上一步得到的图像规模一致，但再次卷积之后的图像信息更为抽象，也更接近全局信息；
 
e. Reshape，即将多维特征图转换为特征向量，得到1600维的特征向量；

f. 经过全连接1600×512，得到512维特征向量；

g. 再次全连接512×2，得到最终的2维向量[0,1]和[1,0]，分别代表游戏屏幕上的是否点击事件。

可以看出，该模型实现了端到端的学习，输入的是游戏屏幕的截图信息（代码中经过opencv处理），输出的是游戏的动作，即是否点击屏幕。深度学习的强大在于其数据拟合能力，不需要传统机器学习中复杂的特征提取过程，而是依靠模型发现数据内部的关系。

## 三、算法：Deep Q Network

有了卷积神经网络模型，那么怎样训练模型？使得模型收敛，从而能够指导游戏动作呢？机器学习分为监督学习、非监督学习和强化学习，这里要介绍的Q Network属于强化学习（Reinforcement Learning）的范畴。在正式介绍Q Network之前，先简单说下它的光荣历史。
2014年Google 4亿美金收购DeepMind的桥段，大家可能听说过。那么，DeepMind是如何被Google给盯上的呢？最终原因可以归咎为这篇论文：
<br>
>Playing Atari with Deep Reinforcement Learning

<br>DeepMind团队通过强化学习，完成了20多种游戏，实现了端到端的学习。其用到的算法就是Q Network。2015年，DeepMind团队在《Nature》上发表了一篇升级版：
<br>
>Human-level control through deep reinforcement learning
<br>
自此，在这类游戏领域，人已经无法超过机器了。后来又有了AlphaGo，以及Master，当然，这都是后话了。其实本文也属于上述论文的范畴，只不过基于TensorFlow平台进行了实现，加入了一些笔者自己的理解而已。
回到正题，Q Network属于强化学习，那么先介绍下强化学习。<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/dqn_netc.png)<br>

>这张图是从UCL的课程中拷出来的，课程链接地址（YouTube）：https://www.youtube.com/watch?v=2pWv7GOvuf0
<br>强化学习过程有两个组成部分：<br>

* 1、智能代理（学习系统）<br>
* 2、环境 <br>
如图所示，在每步迭代过程中，首先智能代理（学习系统）接收环境的状态`st`，然后产生动作`at`作用于环境，环境接收动作`at`，并且对其进行评价，反馈给智能代理`rt`。不断的循环这个过程，就会产生一个状态/动作/反馈的序列：（s1, a1, r1, s2, a2, r2.....,sn, an, rn），而这个序列让我们很自然的想起了:

* 马尔科夫决策过程<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/maer.png)
<br>马尔科夫决策过程与著名的HMM（隐马尔科夫模型）相同的是，它们都具有马尔科夫特性。那么什么是马尔科夫特性呢？简单来说，就是未来的状态只取决于当前的状态，与过去的状态无关。

> HMM（马尔科夫模型）在语音识别，行为识别等机器学习领域有较为广泛的应用。条件随机场模型（Conditional Random Field）则用于自然语言处理。两大模型是语音识别、自然语言处理领域的基石。
<br>
上图可以用一个很形象的例子来说明。比如你毕业进入了一个公司，你的初始职级是T1（对应图中的 s1），你在工作上刻苦努力，追求上进（对应图中的a1），然后领导觉得你不错，准备给你升职（对应图中的r1），于是，你升到了T2；你继续刻苦努力，追求上进......不断的努力，不断的升职，最后升到了sn。当然，你也有可能不努力上进，这也是一种动作，换句话说，该动作a也属于动作集合A，然后得到的反馈r就是没有升职加薪的机会。

这里注意下，我们当然希望获取最多的升职，那么问题转换为：如何根据当前状态s（s属于状态集S），从A中选取动作a执行于环境，从而获取最多的r，即r1 + r2 ……+rn的和最大 ？这里必须要引入一个数学公式：状态值函数。<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/state.png)
<br>公式中有个折合因子γ，其取值范围为[0,1]，当其为0时，表示只考虑当前动作对当前的影响，不考虑对后续步骤的影响，当其为1时，表示当前动作对后续每步都有均等的影响。当然，实际情况通常是当前动作对后续得分有一定的影响，但随着步数增加，其影响减小。
从公式中可以看出，状态值函数可以通过迭代的方式来求解。增强学习的目的就是求解马尔可夫决策过程（MDP）的最优策略。
<br>
> 策略就是如何根据环境选取动作来执行的依据。策略分为稳定的策略和不稳定的策略，稳定的策略在相同的环境下，总是会给出相同的动作，不稳定的策略则反之，这里我们主要讨论稳定的策略。
<br>
求解上述状态函数需要采用动态规划的方法，而具体到公式，不得不提：

* 贝尔曼方程<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/belman.png)
<br>其中，π代表上述提到的策略，Q π (s, a)相比于V π (s)，引入了动作，被称作动作值函数。对贝尔曼方程求最优解，就得到了贝尔曼最优性方程。<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/beerm1.png)
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/beerm2.png)
<br>求解该方程有两种方法：策略迭代和值迭代。

* 策略迭代
策略迭代分为两个步骤：策略评估和策略改进，即首先评估策略，得到状态值函数，其次，改进策略，如果新的策略比之前好，就替代老的策略。<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/policy1.png)<br>
* 值迭代
从上面我们可以看到，策略迭代算法包含了一个策略估计的过程，而策略估计则需要扫描(sweep)所有的状态若干次，其中巨大的计算量直接影响了策略迭代算法的效率。而值迭代每次只扫描一次，更新过程如下：<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/policy2.png)
<br>即在值迭代的第k+1次迭代时，直接将能获得的最大的Vπ(s)值赋给Vk+1。

* Q-Learning
Q-Learning是根据值迭代的思路来进行学习的。该算法中，Q值更新的方法如下：<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/ql.png)
<br>虽然根据值迭代计算出目标Q值，但是这里并没有直接将这个Q值（是估计值）直接赋予新的Q，而是采用渐进的方式类似梯度下降，朝目标迈近一小步，取决于α，这就能够减少估计误差造成的影响。类似随机梯度下降，最后可以收敛到最优的Q值。具体算法如下：<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/ql1.png)
<br>如果没有接触过动态规划的童鞋看上述公式可能有点头大，下面通过表格来演示下Q值更新的过程，大家就明白了。

状态	a1	a2	a3	a4<br>
s1	Q(1, 1)	Q(1, 2)	Q(1, 3)	Q(1, 4) <br>
s2	Q(2, 1)	Q(2, 2)	Q(2, 3)	Q(2, 4) <br>
s3	Q(3, 1)	Q(3, 2)	Q(3, 3)	Q(3, 4) <br>
s4	Q(4, 1)	Q(4, 2)	Q(4, 3)	Q(4, 4) <br>
Q-Learning算法的过程就是存储Q值的过程。上表中，横列为状态s，纵列为Action a，s和a决定了表中的Q值。

第一步：初始化，将表中的Q值全部置0；
第二步：根据策略及状态s，选择a执行。假定当前状态为s1，由于初始值都为0，所以任意选取a执行，假定这里选取了a2执行，得到了reward为1，并且进入了状态s3。根据Q值更新公式：<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/st1.png)
<br>来更新Q值，这里我们假设α是1，λ也等于1，也就是每一次都把目标Q值赋给Q。那么这里公式变成：<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/st2.png)
<br>所以在这里，就是<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/st3.png)
<br>那么对应的s3状态，最大值是0，所以
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/st4.png)
<br>Q表格就变成：
状态	a1	a2	a3	a4 <br>
s1	0	1	0	0 <br>
s2	0	0	0	0 <br>
s3	0	0	0	0 <br>
s4	0	0	0	0 <br>
然后置位当前状态s为s3。

第三步：继续循环操作，进入下一次动作，当前状态是s3，假设选择动作a3，然后得到reward为2，状态变成s1，那么我们同样进行更新：<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/st5.png)
<br>所以Q的表格就变成：
状态	a1	a2	a3	a4 <br>
s1	0	1	0	0 <br>
s2	0	0	0	0 <br>
s3	0	0	3	0 <br>
s4	0	0	0	0 <br>
第四步： 继续循环，Q值在试验的同时反复更新，直到收敛。
上述表格演示了具有4种状态/4种行为的系统，然而在实际应用中，以本文讲到的Flappy Bird游戏为例，界面为80*80个像素点，每个像素点的色值有256种可能。那么实际的状态总数为256的80*80次方，这是一个很大的数字，直接导致无法通过表格的思路进行计算。因此，为了实现降维，这里引入了一个价值函数近似的方法，通过一个函数表近似表达价值函数：<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/st6.png)
<br>其中，ω 与 b 分别为参数。看到这里，终于可以联系到前面提到的神经网络了，上面的表达式不就是神经元的函数吗？

* Q-network
下面这张图来自论文《Human-level Control through Deep Reinforcement Learning》，其中详细介绍了上述将Q值神经网络化的过程。（感兴趣的可以点之前的链接了解原文～）<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/Big.png)
<br>以本文为例，输入是经过处理的4个连续的80x80图像，然后经过三个卷积层，一个池化层，两个全连接层，最后输出包含每一个动作Q值的向量。
现在已经将Q-learning神经网络化为Q-network了，接下来的问题是如何训练这个神经网络。神经网络训练的过程其实就是一个最优化方程求解的过程，定义系统的损失函数，然后让损失函数最小化的过程。
训练过程依赖于上述提到的DQN算法，以目标Q值作为标签，因此，损失函数可以定义为：<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/dqnb.png)
上面公式是s'，a'即下一个状态和动作。确定了损失函数，确定了获取样本的方式，DQN的整个算法也就成型了！<br>
![](https://github.com/2411148291/DeepLearningFlappyBird/blob/master/images/last.png)
<br>值得注意的是这里的D—Experience Replay，也就是经验池，就是如何存储样本及采样的问题。
由于玩Flappy Bird游戏，采集的样本是一个时间序列，样本之间具有连续性，如果每次得到样本就更新Q值，受样本分布影响，效果会不好。因此，一个很直接的想法就是把样本先存起来，然后随机采样如何？这就是Experience Replay的思想。
算法实现上，先反复实验，并且将实验数据存储在D中；存储到一定程度，就从中随机抽取数据，对损失函数进行梯度下降。

## 四、代码：TensorFlow实现
代码从结构上来讲，主要分为以下几部分：
GameState游戏类，frame_step方法控制移动
CNN模型构建
OpenCV-Python图像预处理方法
模型训练过程

### 1. GameState游戏类及frame_step方法
通过Python实现游戏必然要用pygame库，其包含时钟、基本的显示控制、各种游戏控件、触发事件等，对此有兴趣的，可以详细了解pygame。frame_step方法的入参为shape为 (2,) 的ndarray，值域： [1,0]：什么都不做； [0,1]：提升Bird。来看下代码实现：
```Python
if input_actions[1] == 1:
    if self.playery > -2 * PLAYER_HEIGHT:
        self.playerVelY = self.playerFlapAcc
        self.playerFlapped = True
        # SOUNDS['wing'].play()
```
后续操作包括检查得分、设置界面、检查是否碰撞等，这里不再详细展开。
frame_step方法的返回值是：
```Python
return image_data, reward, terminal
```
分别表示界面图像数据，得分以及是否结束游戏。对应前面强化学习模型，界面图像数据表示环境状态 s，得分表示环境给予学习系统的反馈 r。

### 2. CNN模型构建

该Demo中包含三个卷积层，一个池化层，两个全连接层，最后输出包含每一个动作Q值的向量。因此，首先定义权重、偏置、卷积和池化函数：
``` Python
# 权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# 偏置
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# 卷积
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

# 池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
```
然后，通过上述函数构建卷积神经网络模型（对代码中参数不解的，可直接往前翻，看上面那张手画的图）。
```Python
def createNetwork():
    # 第一层卷积
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])
    # 第二层卷积
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    # 第三层卷积
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    # 第一层全连接
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])
    # 第二层全连接
    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    # 输入层
    s = tf.placeholder("float", [None, 80, 80, 4])
    # 第一层隐藏层+池化层
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # 第二层隐藏层（这里只用了一层池化层）
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    # h_pool2 = max_pool_2x2(h_conv2)
    # 第三层隐藏层
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)
    # Reshape
    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    # 全连接层
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
    # 输出层
    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1
```
### 3. OpenCV-Python图像预处理方法

>本人在win10的Anaconda中安装的opencv3，命令 conda install -c https://conda.anaconda.org/menpo opencv3 过程中遇到http和URL错误，保证你的机器能访问到上述https网址，最终安装成功。建议安装opencv3。<br>
这部分主要对frame_step方法返回的数据进行了灰度化和二值化，也就是最基本的图像预处理方法。
```python
x_t, r_0, terminal = game_state.frame_step(do_nothing)
# 首先将图像转换为80*80，然后进行灰度化
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
# 对灰度图像二值化
ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
# 四通道输入图像
s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
```
### 4. DQN训练过程

这是代码部分要讲的重点，也是上述Q-learning算法的代码化。

i. 在进入训练之前，首先创建一些变量：
```python
# define the cost function
a = tf.placeholder("float", [None, ACTIONS])
y = tf.placeholder("float", [None])
readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)
cost = tf.reduce_mean(tf.square(y - readout_action))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

# open up a game state to communicate with emulator
game_state = game.GameState()

# store the previous observations in replay memory
D = deque()
```
在TensorFlow中，通常有三种读取数据的方式：Feeding、Reading from files和Preloaded data。Feeding是最常用也最有效的方法。即在模型（Graph）构建之前，先使用placeholder进行占位，但此时并没有训练数据，训练是通过feed_dict传入数据。

这里的`a`表示输出的动作，即强化学习模型中的Action，`y`表示标签值，`readout_action`表示模型输出与`a`相乘后，在一维求和，损失函数对标签值与输出值的差进行平方，`train_step`表示对损失函数进行`Adam`优化。

赋值的过程为：
```python
# perform gradient step
train_step.run(feed_dict={
    y: y_batch,
    a: a_batch,
    s: s_j_batch}
)
```
ii. 创建游戏及经验池 D
```python
# open up a game state to communicate with emulator
game_state = game.GameState()

# store the previous observations in replay memory
D = deque()
```
经验池 D采用了队列的数据结构，是TensorFlow中最基础的数据结构，可以通过`dequeue()`和`enqueue([y])`方法进行取出和压入数据。经验池 D用来存储实验过程中的数据，后面的训练过程会从中随机取出一定量的batch进行训练。

变量创建完成之后，需要调用TensorFlow系统方法tf.global_variables_initializer()添加一个操作实现变量初始化。运行时机是在模型构建完成，Session建立之初。比如：
```python
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
...
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Later, when launching the model
with tf.Session() as sess:
  # Run the init operation.
  sess.run(init_op)
  ...
  # Use the model
  ...
 ```
iii. 参数保存及加载

采用TensorFlow训练模型，需要将训练得到的参数进行保存，不然一关机，就一夜回到解放前了。TensorFlow采用Saver来保存。一般在Session()建立之前，通过`tf.train.Saver()`获取Saver实例。
```python
saver = tf.train.Saver()
```
变量的恢复使用`saver`的`restore`方法：
```python
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...
 ```
在该Demo训练时，也采用了Saver进行参数保存。
```python
# saving and loading networks
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state("saved_networks")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")
 ```
首先加载CheckPointState文件，然后采用`saver.restore`对已存在参数进行恢复。
在该Demo中，每隔10000步，就对参数进行保存：
```python
# save progress every 10000 iterations
if t % 10000 == 0:
    saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)
```
iv. 实验及样本存储

首先，根据ε 概率选择一个Action。
```python
# choose an action epsilon greedily
readout_t = readout.eval(feed_dict={s: [s_t]})[0]
a_t = np.zeros([ACTIONS])
action_index = 0
if t % FRAME_PER_ACTION == 0:
    if random.random() <= epsilon:
        print("----------Random Action----------")
        action_index = random.randrange(ACTIONS)
        a_t[random.randrange(ACTIONS)] = 1
    else:
        action_index = np.argmax(readout_t)
        a_t[action_index] = 1
else:
    a_t[0] = 1  # do nothing
```
这里，`readout_t`是训练数据为之前提到的四通道图像的模型输出。`a_t`是根据ε 概率选择的Action。

其次，执行选择的动作，并保存返回的状态、得分。
```python
# run the selected action and observe next state and reward
x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
x_t1 = np.reshape(x_t1, (80, 80, 1))
# s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

# store the transition in D
D.append((s_t, a_t, r_t, s_t1, terminal))
```
经验池`D`保存的是一个马尔科夫序列。`(s_t, a_t, r_t, s_t1, terminal)`分别表示`t`时的状态`s_t`，执行的动作`a_t`，得到的反馈`r_t`，以及得到的下一步的状态`s_t1`和游戏是否结束的标志`terminal`。

在下一训练过程中，更新当前状态及步数：
```python
# update the old values
s_t = s_t1
t += 1
```
重复上述过程，实现反复实验及样本存储。

v. 通过梯度下降进行模型训练

在实验一段时间后，经验池`D`中已经保存了一些样本数据后，就可以从这些样本数据中随机抽样，进行模型训练了。这里设置样本数为`OBSERVE = 100000`.。随机抽样的样本数为`BATCH = 32`。
```python
if t > OBSERVE:
    # sample a minibatch to train on
    minibatch = random.sample(D, BATCH)

    # get the batch variables
    s_j_batch = [d[0] for d in minibatch]
    a_batch = [d[1] for d in minibatch]
    r_batch = [d[2] for d in minibatch]
    s_j1_batch = [d[3] for d in minibatch]

    y_batch = []
    readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
    for i in range(0, len(minibatch)):
        terminal = minibatch[i][4]
        # if terminal, only equals reward
        if terminal:
            y_batch.append(r_batch[i])
        else:
            y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

    # perform gradient step
    train_step.run(feed_dict={
        y: y_batch,
        a: a_batch,
        s: s_j_batch}
    )
```
`s_j_batch`、`a_batch`、`r_batch`、`s_j1_batch`是从经验池`D`中提取到的马尔科夫序列（Java童鞋羡慕Python的列表推导式啊），`y_batch`为标签值，若游戏结束，则不存在下一步中状态对应的`Q`值（回忆Q值更新过程），直接添加`r_batch`，若未结束，则用折合因子（0.99）和下一步中状态的最大Q值的乘积，添加至`y_batch`。
最后，执行梯度下降训练，`train_step`的入参是`s_j_batch`、`a_batch`和`y_batch`。差不多经过2000000步（在本机上大概10个小时）训练之后，就能达到本文开头动图中的效果啦。

## 声明
本篇内容主要记录学习的过程，大多内容来自 Young的 http://www.cnblogs.com/younghao/p/6696739.html 的分析。
