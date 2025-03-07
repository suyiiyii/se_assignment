# 第一次个人编程作业

## **需求**

题目：论文查重

描述如下：

设计一个论文查重算法，给出一个原文文件和一个在这份原文上经过了增删改的抄袭版论文的文件，在答案文件中输出其重复率。

- 原文示例：今天是星期天，天气晴，今天晚上我要去看电影。
- 抄袭版示例：今天是周天，天气晴朗，我晚上要去看电影。

要求输入输出采用文件输入输出，规范如下：

- 从**命令行参数**给出：论文原文的文件的**绝对路径**。
- 从**命令行参数**给出：抄袭版论文的文件的**绝对路径**。
- 从**命令行参数**给出：输出的答案文件的**绝对路径**。

我们提供一份样例，课堂上下发，上传到班级群，使用方法是：orig.txt 是原文，其他 orig_add.txt 等均为抄袭版论文。

注意：答案文件中输出的答案为浮点型，精确到小数点后两位

## 迭代版本

### TF-IDF&余弦相似度

计算文本的 TF-IDF 向量，然后计算两个向量的余弦相似度。

#### 优点

- 简单，计算快
- 可以应对简单的顺序替换

#### 不足

- 无法处理语义相似的文本（同义词）

### 使用 jieba 分词，配合文本向量模型计算相似度

使用 jieba 分词，将文本分成单词，然后使用文本向量模型计算相似度。

#### 优点

- 可以处理语义相似的文本（同义词）

#### 不足

- 计算量较大，速度较慢

## 测试结果

```shell
PS C:\Users\suyiiyii\Documents\git\se_assignment\3123004223\similarity_detection> uv run .\main.py
正在加载中文词向量模型...
中文词向量模型加载完成
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\suyiiyii\AppData\Local\Temp\jieba.cache
Loading model cost 0.577 seconds.
Prefix dict has been built successfully.
orig_0.8_add.txt: 99.59%
orig_0.8_del.txt: 99.79%
orig_0.8_dis_1.txt: 99.97%
orig_0.8_dis_10.txt: 99.86%
orig_0.8_dis_15.txt: 99.55%
```

可见效果非常显著
