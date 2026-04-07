# MiniMind-Learning
Reproduce the Transformer architecture.

## 项目简介
跟随MiniMind项目，从零理解Transformer和LLM底层原理

## 学习进度
- [ ] Transformer架构理解
- [ ] Self-Attention机制
- [ ] 位置编码
- [ ] 训练流程


首先教程来自b站【【2025/Minimind】Only三小时！Pytorch从零手敲大模型，架构到训练全教程】 https://www.bilibili.com/video/BV1T2k6BaEeC/?p=4&share_source=copy_web&vd_source=567a5b401b6e2f94758f0d0e37e9044f
记录一下这两个月来的自学成果。

## 每日笔记
### Day 1
今天理解了：...
遇到的问题：...
怎么解决的：...

神经网络是什么 
attention是什么
---
 
## 学习进度
 
- [x] 神经网络基础（CNN、卷积层、池化层、全连接层）
- [x] 反向传播与优化器（SGD、Adam）
- [x] Attention 机制原理
- [x] PyTorch 框架理解
- [ ] Transformer 架构代码精读
- [ ] Self-Attention 手动实现
- [ ] 位置编码（Positional Encoding）
- [ ] 训练流程跑通
- [ ] 实验对比记录
 
---
 
## 核心概念笔记
 
### 1. 神经网络与 CNN
 
图片在计算机中表示为像素矩阵，例如一张彩色图片为 `224 × 224 × 3`（长 × 宽 × RGB通道）。
 
CNN 处理流程：
 
```
输入图片 (224×224×3)
  ↓ 卷积层：64个卷积核，每个负责提取一种特征（边缘、纹理等）
(224×224×64)
  ↓ 池化层：每2×2区域取最大值，尺寸减半
(112×112×64)
  ↓ 继续卷积+池化，通道数翻倍，尺寸减半...
(7×7×512)
  ↓ Flatten 展平为一维向量
(25088个数字)
  ↓ 全连接层
(1000个类别概率)
  ↓ Softmax 输出最终预测
```
 
**关键理解：**
- 卷积核数量（64、128...）是人工定义的超参数，核内具体数值由训练自动学习
- 池化层每次尺寸减半，卷积核数量翻倍——这是 VGG 的设计规律，不是硬性规定
- 最后的输出类别数（如1000）也是人工定义，取决于任务有多少类别
 
---
 
### 2. 反向传播与优化器
 
**训练流程：**
 
```
① 正向传播：图片 → 网络 → 输出预测值
② 损失函数：预测值 vs 正确答案 → 计算误差
   - 分类任务：交叉熵（Cross Entropy）
   - 回归任务：均方误差（MSE）
③ 反向传播：误差从后往前传播，计算每层梯度
④ 优化器：根据梯度更新参数
⑤ 重复以上步骤直到模型收敛
```
 
> 反向传播不是"结果不好才用"，而是**每一次训练迭代都会发生**。
 
**优化器对比：**
 
| 优化器 | 特点 |
|--------|------|
| SGD | 固定步长，简单但容易震荡，大模型图像任务仍在使用 |
| Adam | 自适应学习率 + 动量，收敛快，大多数任务默认首选 |
 
**局部最优问题：** 梯度下降只看当前位置的坡度，可能陷入局部最低点。实践中通过随机初始化、学习率调度、Dropout 等方式缓解。高维参数空间中，真正的局部最优点反而很少见。
 
---
 
### 3. Attention 机制
 
RNN 的问题：按顺序读句子，容易"忘记"前面的内容。
 
**Attention 的核心思想：**
 
> 理解一个词时，同时看整句话，对不同的词给予不同的注意力权重。
 
```
句子："The cat sat on the mat because it was tired"
 
理解 "it" 时的注意力权重：
The(0.01) cat(0.85) sat(0.03) on(0.01) mat(0.02) it(0.05) tired(0.03)
 
"it" 的最终表示 = 0.85×cat的特征 + 0.03×sat的特征 + ...
```
 
**Self-Attention：** 句子中每个词同时对其他所有词做上述计算，并行处理，不用按顺序读。这是 Transformer 比 RNN 快且强的核心原因。
 
---
 
### 4. PyTorch 是什么
 
```
sklearn（机器学习）= 买成品家具，直接调用 RF、KNN 等现成模型
PyTorch（深度学习）= 买木板和工具，自己设计组装神经网络
```
 
**PyTorch 四层架构：**
 
```
Tensor（张量）     → 多维数组，一切的基础，可放 GPU 运算
Autograd（自动求导）→ 自动计算梯度，loss.backward() 一行搞定
torch.nn           → 封装好的积木：Conv2d、Linear、Embedding...
训练工具           → optim（优化器）、DataLoader（数据加载）
```
 
---
 
## 每日学习记录
 
### Day 1
**今天理解了：** CNN 的完整处理流程，从像素矩阵到最终分类结果；反向传播是训练中每步都发生的过程；Attention 机制的本质是加权求和；PyTorch 和 sklearn 的定位区别。
 
**遇到的问题：** 暂无
 
**明日计划：** 开始精读 MiniMind 源码，从 model.py 入手
 
---
 
## 参考资料
 
- [MiniMind 原项目](https://github.com/jingyaogong/minimind)
- [Attention Is All You Need 论文](https://arxiv.org/abs/1706.03762)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
 
