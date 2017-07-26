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

c. 再次卷积，卷积核为4×4×32×64，步距为2，得到图像5×5×64；神经元个数：5×5×64=1600个

d. 再次卷积，卷积核为3×3×64*64，步距为2，得到图像5×5×64，虽然与上一步得到的图像规模一致，但再次卷积之后的图像信息更为抽象，也更接近全局信息；

e. Reshape，即将多维特征图转换为特征向量，得到1600维的特征向量；

f. 经过全连接1600×512，得到512维特征向量；

g. 再次全连接512×2，得到最终的2维向量[0,1]和[1,0]，分别代表游戏屏幕上的是否点击事件。

可以看出，该模型实现了端到端的学习，输入的是游戏屏幕的截图信息（代码中经过opencv处理），输出的是游戏的动作，即是否点击屏幕。深度学习的强大在于其数据拟合能力，不需要传统机器学习中复杂的特征提取过程，而是依靠模型发现数据内部的关系。

# 四、代码：TensorFlow实现
代码从结构上来讲，主要分为以下几部分：
GameState游戏类，frame_step方法控制移动
CNN模型构建
OpenCV-Python图像预处理方法
模型训练过程

1. GameState游戏类及frame_step方法
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

2. CNN模型构建

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
## 3. OpenCV-Python图像预处理方法

在Ubuntu中安装opencv的步骤比较麻烦，当时也踩了不少坑，各种Google解决。建议安装opencv3。
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
## 4. DQN训练过程

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

这里的a表示输出的动作，即强化学习模型中的Action，y表示标签值，readout_action表示模型输出与a相乘后，在一维求和，损失函数对标签值与输出值的差进行平方，train_step表示对损失函数进行Adam优化。

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
经验池 D采用了队列的数据结构，是TensorFlow中最基础的数据结构，可以通过dequeue()和enqueue([y])方法进行取出和压入数据。经验池 D用来存储实验过程中的数据，后面的训练过程会从中随机取出一定量的batch进行训练。

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

采用TensorFlow训练模型，需要将训练得到的参数进行保存，不然一关机，就一夜回到解放前了。TensorFlow采用Saver来保存。一般在Session()建立之前，通过tf.train.Saver()获取Saver实例。
```python
saver = tf.train.Saver()
```
变量的恢复使用saver的restore方法：
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
首先加载CheckPointState文件，然后采用saver.restore对已存在参数进行恢复。
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
这里，readout_t是训练数据为之前提到的四通道图像的模型输出。a_t是根据ε 概率选择的Action。

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
经验池D保存的是一个马尔科夫序列。(s_t, a_t, r_t, s_t1, terminal)分别表示t时的状态s_t，执行的动作a_t，得到的反馈r_t，以及得到的下一步的状态s_t1和游戏是否结束的标志terminal。

在下一训练过程中，更新当前状态及步数：
```python
# update the old values
s_t = s_t1
t += 1
```
重复上述过程，实现反复实验及样本存储。

v. 通过梯度下降进行模型训练

在实验一段时间后，经验池D中已经保存了一些样本数据后，就可以从这些样本数据中随机抽样，进行模型训练了。这里设置样本数为OBSERVE = 100000.。随机抽样的样本数为BATCH = 32。
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
s_j_batch、a_batch、r_batch、s_j1_batch是从经验池D中提取到的马尔科夫序列（Java童鞋羡慕Python的列表推导式啊），y_batch为标签值，若游戏结束，则不存在下一步中状态对应的Q值（回忆Q值更新过程），直接添加r_batch，若未结束，则用折合因子（0.99）和下一步中状态的最大Q值的乘积，添加至y_batch。
最后，执行梯度下降训练，train_step的入参是s_j_batch、a_batch和y_batch。差不多经过2000000步（在本机上大概10个小时）训练之后，就能达到本文开头动图中的效果啦。
