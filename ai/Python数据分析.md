# 常用数据分析开源库

1. NumPy：一个运行速度非常快的数学库，主要用于数组计算，处理N维数组对象：ndarray。

2. Pandas：分析结构化数据的工具集，基础是NumPy。用于数据挖掘、分析、清洗。

3. 1. Series：类似于一维数组的对象。
   2. DataFrame：表格型的数据结构。

4. Matplotlib：功能强大的数据可视化开源Python库

5. Seaborn：Python数据可视化开源库, 建立在matplotlib之上，并集成了pandas的数据结构。面向数据集的API，与Pandas配合使用起来比直接使用Matplotlib更方便

6. Sklearn：scikit-learn 是基于 Python 语言的机器学习工具：建立在 NumPy ，SciPy 和 matplotlib 上

7. jupyter notebook：是一个开源Web应用程序, 可以创建和共享代码、公式、可视化图表、笔记文档；是数据分析学习和开发的首选开发环境。

 

# Python数据分析环境搭建

## Anaconda介绍

1. 是最流行的数据分析平台，全球两千多万人在使用
2. 附带了一大批常用数据科学包
3. 包含了虚拟环境管理工具

 

## Anaconda的命令操作

| 命令                                      | 作用             |
| ----------------------------------------- | ---------------- |
| conda install 包名字                      | 安装包           |
| pip install 报名字                        | 安装包           |
| conda create -n 虚拟环境名字              | 创建虚拟环境     |
| conda activate 虚拟环境名字               | 进入虚拟环境     |
| conda deactivate 虚拟环境名字             | 退出虚拟环境     |
| conda remove -n 虚拟环境名字 --all        | 删除虚拟环境     |
| conda env list                            | 查看所有虚拟环境 |
| conda create -n new_name --clone old_name | 克隆一个环境     |

使用pip时最好指定安装源，加快安装包下载地址：

```
#通过阿里云镜像安装
pip install 包名 -i https://mirrors.aliyun.com/pypi/simple/  
```

 

## Jupyter NoteBook使用

1. Shift+Enter，执行本单元代码，并跳转到下一单元
2. Ctrl+Enter，执行本单元代码，留在本单元

 

ESC进入命令模式，操作：

1. `Y`，cell切换到Code模式
2. `M`，cell切换到Markdown模式，需要执行cell代码才能显示渲染后的结果。
3.  `A`，在当前cell的上面添加cell
4.  `B`，在当前cell的下面添加cell
5.   `双击D`：删除当前cell

 

编辑模式：按Enter进入，或鼠标点击代码编辑框体的输入区域，操作：

1. 撤销：`Ctrl+Z`（Mac:CMD+Z）
2. 反撤销: `Ctrl + Y`（Mac:CMD+Y）
3. 补全代码：变量、方法后跟`Tab键`
4. 为一行或多行代码添加/取消注释：`Ctrl+/`（Mac:CMD+/）
5. 代码提示: `shift + Tab`

 

## PyCharm连接Anaconda

设置，python解释器选择Anaconda

 

![img](https://uploader.shimo.im/f/iXvTfthuaPRW8ntp.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDM2MjMsImZpbGVHVUlEIjoiUjEzamRMSnlybElKMDJrNSIsImlhdCI6MTc1OTgwMzMyMywiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.HIe_EilEEwOoEWvoiufkIs1hKoD3UvSds9WeV0tOESk)

 

Anaconda界面启动Jupyter：

![img](https://uploader.shimo.im/f/bVVe5anSeqCzR8X1.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDM2MjMsImZpbGVHVUlEIjoiUjEzamRMSnlybElKMDJrNSIsImlhdCI6MTc1OTgwMzMyMywiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.HIe_EilEEwOoEWvoiufkIs1hKoD3UvSds9WeV0tOESk)

 

 

创建Jupyter笔记：

![img](https://uploader.shimo.im/f/jQ2YIxDtOdCwbs0F.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDM2MjMsImZpbGVHVUlEIjoiUjEzamRMSnlybElKMDJrNSIsImlhdCI6MTc1OTgwMzMyMywiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.HIe_EilEEwOoEWvoiufkIs1hKoD3UvSds9WeV0tOESk)

 

 

连接远程Jupyter：

![img](https://uploader.shimo.im/f/yghjUEY87z9BPnLA.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDM2MjMsImZpbGVHVUlEIjoiUjEzamRMSnlybElKMDJrNSIsImlhdCI6MTc1OTgwMzMyMywiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.HIe_EilEEwOoEWvoiufkIs1hKoD3UvSds9WeV0tOESk)

 

# numpy

 

## 介绍

NumPy的出现一定程度上解决了Python运算性能不佳的问题，同时提供了更加精确的数据类型，使其具备了构造复杂数据类型的能力。

1. 本身是由C语言开发，是个很基础的扩展，NumPy被Python其它科学计算包作为基础包，因此理解np的数据类型对python数据分析十分重要。
2. NumPy重在数值计算，主要用于多维数组（矩阵）处理的库。用来存储和处理大型矩阵，比Python自身的嵌套列表结构要高效的多

 

 

NumPy的数组类被称作ndarray，通常被称作数组。

1. ndarray数组是一个多维的数组对象（矩阵），称ndarray(N-Dimensional Array)
2. 具有矢量算术运算能力和复杂的广播能力，并具有执行速度快和节省空间的特点
3. 注意：ndarray的下标从0开始，且数组里的所有元素必须是相同类型。

 

## 基本属性

```
import numpy as np


# 创建np数组，0- 14
arr = np.arange(15)
print(arr)


# 创建一个二维数组。数组3个元素，每个元素师5个元素的一维数组。
#可以这么理解，将元素切成3份，每份5个元素
arr = arr.reshape(3, 5)
print(arr)


print(f"数组的维度: {arr.shape}") # (3, 5)
print(f"数组轴的个数：{arr.ndim}") # 是几位数组 ：2
print(f"数组元素类型：{arr.dtype}") # int64
print(f"数组元素个数：{arr.size}") # 15，数组压平后总元素个数
print(f"数组类型：{type(arr)}") # <class 'numpy.ndarray'>
```

 

## ndarray的创建

```
import numpy as np 


#基于数组创建
arr = np.array([1, 2, 3, 4])
print(arr) # [1 2 3 4]


# zeros创建一个全是0的数组
# 创建2个一维数组，每个数组3个元素
zero = np.zeros((2, 3))
print(zero) # [[0. 0. 0.] 0. 0. 0.]]


# 最外层数组，2个元素 [ [], [] ]
# 第二层数组3个元素: [ [ [], [], [] ], [ [], [], []]]
# 第三层4个元素 [ 
# [ [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0] ], 
# [ [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
# ]
zero1 = np.zeros((2,3,4))
print(zero1) 


# 创建全是1的数组
ones = np.ones((2,3))
print(ones)


# 创建一个基于内存数据的数组，将内存当前数据解析为整数
empty = np.empty((2, 3))
print(empty)


# arange  起始, 结束, 步长, 类型
range = np.arange(10, 20, 3,dtype=int)
print(range)




# matrix 是 ndarray 的子类，只能生成 2 维的矩阵


# 基于模式创建
# 1 2;3 也可以识别
m1 = np.mat("1,2;3,4")
print(m1) # [ [1 2] [3 4] ]


#基于数组创建
m2 = np.matrix([ [1, 2], [3, 4]])
print(m2)
```

 

## 创建随机数矩阵

```
# 创建随机数矩阵
import numpy as np


# 生成指定维度的随机数，浮点型，数值访问：[0,1)
arr = np.random.rand(3, 4)
print(arr)


# 指定区间, [start, end), size指定维度
arr = np.random.randint(1, 5, size=(3,4))
print(arr)


# 生成均匀分布的样本值(浮点数),  size指定维度
arr = np.random.uniform(-1, 5, size=(3, 4))
print(arr)
```

 

## 数列

```
import numpy as np


# 创建等比数列
# 0, 10, 10：开始位置为2的0次幂，结束位置为10的2次幂。创建10个元素，元素之前是等比数列
# base 设置基数
arr = np.logspace(0, 10, 10,base=2)
print(arr)


# 创建等差数列 第一个参数表示起始点，第二个参数表示终止点，第三个参数表示数列的个数。
arr1 = np.linspace(1, 10, 2)
print(arr)
```

 

## 基本函数

```
  np.ceil(): 向上最接近的整数，参数是 number 或 array
  np.floor(): 向下最接近的整数，参数是 number 或 array
  np.rint(): 四舍五入，参数是 number 或 array
  np.isnan(): 判断元素是否为 NaN(Not a Number)，参数是 number 或 array
  np.multiply(): 元素相乘，参数是 number 或 array
  np.divide(): 元素相除，参数是 number 或 array
  np.abs()：元素的绝对值，参数是 number 或 array
  np.where(condition, x, y): 三元运算符，x if condition else y
```

 

multiply/divide 如果是两个ndarray进行运算 shape必须一致

 

```
import numpy as np
arr = np.random.random(10)
print(arr)


# 对矩阵所有元素执行函数，返回新的矩阵，不修改原来的数据
print(np.ceil(arr))
print(arr)
```

## 统计函数

```
  np.mean(), np.sum()：所有元素的平均值，所有元素的和，参数是 number 或 array
  np.max(), np.min()：所有元素的最大值，所有元素的最小值，参数是 number 或 array
  np.std(), np.var()：所有元素的标准差，所有元素的方差，参数是 number 或 array
  np.argmax(), np.argmin()：最大值的下标索引值，最小值的下标索引值，参数是 number 或 array
  np.cumsum(), np.cumprod()：返回一个一维数组，每个元素都是之前所有元素的 累加和 和 累乘积，参数是 number 或 array
   # 多维数组默认统计全部维度，axis参数可以按指定轴心统计，值为0则按列统计，值为1则按行统计。
```

 

```
  arr = np.arange(12).reshape(3, 4)
  print(arr)
  print(np.cumsum(arr))   # 返回一个一维数组, 每个元素都是之前所有元素的 累加和
  print(np.sum(arr))      # 所有元素的和
  print(np.sum(arr, axis = 0))  #数组的按列统计和
  print(np.sum(arr, axis = 1))  #数组的按行统计和
```

## 比较、去重、排序

```
## 比较函数
arr = np.random.randint(-5, 10, 10)
print(arr)
# 至少有一个元素满足指定条件，返回True
print(np.any(arr > 0))
# np.all(): 所有的元素满足指定条件，返回True
print(np.all(arr > 0))




## 去重函数
#np.unique():找到唯一值并返回排序结果，类似于Python的set集合
arr = np.array([[1, 2, 1], [2, 3, 4]])
print(arr)
print(np.unique(arr)) # [1 2 3 4]，返回去重并且排序一维数组。




## 排序函数


# sort排序
arr = np.array([1, 2, 34, 5])
print("原数组arr:", arr)
# np.sort ，不改变原数据
sort_arr = np.sort(arr)
print("np sort后数组:", arr)


# arr.sort在原数据上进行修改
sort_arr1 = arr.sort()
print("arr sort后数组:", arr)
```

 

## 运算

数组计算时，对应位置的元素进行运算，计算的前提是2个数组的shape相同。计算产生新的数组，shape和原数组相同。

```
import numpy as np


a = np.array([10, 20, 30])
b = np.arange(3)
print("数组a:", a)
print("数组b:", b)
print("数组a + 数组b:", a + b) # 数组a + 数组b: [10 21 32]
```

 

矩阵运算：

```
arr_a.dot(arr_b) 前提 arr_a 列数 = arr_b行数


x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[6, 23], [-1, 7], [8, 9]])
print(x)
print(y)
print(x.dot(y))
print(np.dot(x, y))
```