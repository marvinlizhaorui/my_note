# 机器学习——K近邻算法

### 概述

k近邻是一种基本分类与回归方法.

- 输入:特征向量

- 输出:实例的类别(可取多类)

- 核心思想:如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，则该样本也属于这个类别，并具有这个类别上样本的特性.

- 优点:计算精度高、对异常值不敏感、无数据输入假定

- 缺点:计算复杂度高、空间复杂度高

- 适用范围:数值型和标称型

### 算法流程

> 1. 收集数据
> 2. 准备数据：距离计算所需要的数值，最好是结构化的数据格式
> 3. 分析数据：可以适用任何方法
> 4. 训练算法：此步骤不适用于KNN
> 5. 测试算法：计算错误率
> 6. 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行k-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理

### kNN算法

##### 构造数据集

```python
import numpy as np

def create_data_set():
    """构造数据集"""
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    label = ['A', 'A', 'B', 'B']
    return group, label
```

##### 生成数据集

```python
group, label = create_data_set()
```

##### 实现kNN算法伪代码

对未知类别属性的数据集种的每个点依次执行以下步骤： 

 
1\. 计算已知类别属性的数据集中的每个点与当前点之间的距离 

 
2\. 按照距离递增次序排序 

 
3\. 选取与当前点距离最小的k个点 

 
4\. 确定前k个点所在类别的出现频率 

 
5\. 返回前k个点出现频率最高的类别作为当前点的预测分类  
​

```python
import operator

def classify0(inX, data_set, label, k):
    """
    KNN算法
    :param inX: 用于分类的输入向量
    :param data_set: 训练样本集
    :param label: 训练标签向量
    :param k: 选取最近邻居的数量
    :return: k个邻居里频率最高的分类
    """

    """距离计算"""
    # 获得样本量
    data_set_size = data_set.shape[0]
    # tile：在行方向重复inX,dataSetSize次，在列方向上重复inX,1次
    diff_mat = np.tile(inX, (data_set_size, 1)) - data_set
    # 离原点的距离，相见后平方
    sq_diff_mat = diff_mat ** 2
    # x轴和y轴差的平方和
    sq_distances = sq_diff_mat.sum(axis=1)
    # 然后开方
    distances = sq_distances ** 0.5
    # argsort函数返回的是数组值从小到大的索引值
    sorted_distance_index = distances.argsort()
    class_count = {}
    """选择距离最小的点"""
    for i in range(k):
        # 返回距离最近的第i个样本所对应的标签
        vote_label = label[sorted_distance_index[i]]
        # print(voteIlabel)
        # print(classCount.get(voteIlabel, 0))
        # 这里的0是设置默认值为0,而代替None。而代码是给出现情况增加次数，出现一次+1
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
        # print(classCount)
    """排序"""
    # 导入运算符模块的itemgetter方法，按照第二个元素的次序对元组进行排序，此处的排序为逆序。
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回频率最大的Label
    return sorted_class_count[0][0]
```

##### 添加算法测试代码

```python
classify0([0, 0], group, label, k=3)
```

### 约会网站示例

##### 加载并解析数据

```python
# 将文本记录转换为numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    # readlines把文件所有内容读取出来，组成一个列表，其中一行为一个元素
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)
    # 返回一个用1000行每行3个0填充的数组，形成特征矩阵
    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = []
    for index, line in enumerate(array_of_lines):
        # 去除每行前后的空格
        line = line.strip()
        # 根据\t把每行分隔成由四个元素组成的列表
        list_from_line = line.split('\t')
        # 选取前3个元素，将它们按顺序存储到特征矩阵中
        return_mat[index, :] = list_from_line[0: 3]
        # 将列表的最后一个元素储存到class_label_vector中去，储存的元素值为整型
        class_label_vector.append(int(list_from_line[-1]))
    return return_mat, class_label_vector
```

##### 获得解析数据

```python
dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
```

##### 使用Matplotlib创建散点图

```python
import matplotlib.pyplot as plt

# 创建Figure实例
fig = plt.figure()
# 添加一个子图，返回Axes实例
ax = fig.add_subplot(111)  # 选取最近邻居的数量
# 生成散点图，x轴使用dating_data_mat第二列数据，y轴使用dating_data_mat的第三列数据
# ax.scatter(x=dating_data_mat[:, 1], y=dating_data_mat[:, 2])
# 个性化标记散点图，形状(s)和颜色(c)
ax.scatter(x=dating_data_mat[:, 1], y=dating_data_mat[:, 2], s=15.0 * np.array(dating_labels), c=np.array(dating_labels))
plt.show()
```

##### 归一化特征值

<center>newValue=(oldValue−min)/(max−min)</center>

```python
def auto_num(data_set):
    """
    归一化特征值
    :param data_set: 数据集
    :return 归一化后的数据集， 列的差值范围， 列的最小值
    """
    # 列的最小值
    min_val = data_set.min()
    # 列的最大值
    max_val = data_set.max()
    # 列的差值范围
    range_s = max_val - min_val
    # 构造返回矩阵
    norm_data_set = np.zeros(shape=np.shape(data_set))
    # m = data_set.shape[0]
    # oldValue - min
    norm_data_set = data_set - np.tile(min_val, (data_set.shape[0], 1))
    # (oldValue - min) / (max - min)
    norm_data_set = norm_data_set / np.tile(range_s, (data_set.shape[0], 1))
    return norm_data_set, range_s, min_val
```

##### 归一化测试

```python
normalize_data_set, ranges, min_val = auto_num(dating_data_mat)
print(normalize_data_set)
```

##### 测试算法

```python
def dating_class_test():
    # 选择测试数据量
    ho_ratio = 0.10
    # 解析数据
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    # 归一化数据
    norm_mat, range_s, min_val = auto_num(dating_data_mat)
    # 拆分10%数据作为测试数据
    m = norm_mat.shape[0]  # 总数据量
    num_test_vec = int(m * ho_ratio)  # 测试数据量
    # 错误样本计数
    error_count = 0.0
    # 对测试数据进行分类，并对比检验结果正确率
    for i in range(num_test_vec):
        classifier_result = classify0(  # classifier_result : k个邻居里频率最高的分类
            norm_mat[i, :],  # 用于分类的输入向量(测试数据, : 表示一行内所有元素)
            norm_mat[num_test_vec: m, :],  # 训练样本集(从测试的数据开始到总数据量结束)
            dating_labels[num_test_vec:m],  # 训练标签向量(从测试的数据开始到总数据量结束)
            3  # 选取最近邻居的数量
        )
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print('the total error rate is: %f' % (error_count / float(num_test_vec)))
```

##### 执行测试

```python
dating_class_test()
```

##### 使用算法

```python
def classify_person():
    """
    根据输入指标，通过分类器进行预测喜欢程度
    :return:
    """
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percentage of time spent playing vedio games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    dating_data, dating_labels = file2matrix('datingTestSet2.txt')
    normalize_matrix, ranges, min_val = auto_num(dating_data)
    # 将输入指标，归一化后代入分类器进行预测
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    print(in_arr, min_val, ranges, (in_arr-min_val)/ranges)
    print(ranges)
    classifier_result = classify0((in_arr-min_val)/ranges, normalize_matrix, dating_labels, 3)
    print("You will probably like this person: ", result_list[classifier_result - 1])
```

##### 执行函数

```python
classify_person()
```

##### 输出

```python
percentage of time spent playing vedio games?20
frequent flier miles earned per year?299
liters of ice cream consumed per year?1
You will probably like this person:  in large doses
```



##### sklearn中实现

```python
from sklearn import neighbors

def knn_classify_person():
    """
        根据输入指标，通过分类器进行预测喜欢程度
    :return:
    """
    result_list = np.array(['not at all', 'in small doses', 'in large doses'])
    percent_tats = float(input('percentage of time spent playing vedio games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))
    dating_data, dating_labels = file2matrix('datingTestSet2.txt')
    normalize_matrix, ranges, min_val = auto_num(dating_data)
    # 将输入指标，归一化后代入分类器进行预测
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    # 声明k为3的knn算法,n_neighbors即是邻居数量，默认值为5
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    # 训练算法
    knn.fit(normalize_matrix, dating_labels)
    # 预测
    classifier_result = knn.predict([(in_arr - min_val) / ranges])
    print("You will probably like this person: ", result_list[classifier_result - 1][0])


# 执行函数
knn_classify_person()

```
