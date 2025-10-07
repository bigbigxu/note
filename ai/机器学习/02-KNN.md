# KNN算法思想

K-近邻算法（K Nearest Neighbor，简称KNN）：：如果一个样本在特征空间中的 k 个最相似的样本中的大多数属于某一个类别，则该样本也属于这个类别。

 

K：最个最相似的样本数量，必须选择合适的数值。

K值过小：容易受到异常点的影响。K值的减小就意味着整体模型变得复杂，容易发生过拟合

K值过大：受到样本均衡的问题，且K值的增大就意味着整体的模型变得简单，欠拟合

 

# KNN算法

解决的问题：分类、回归问题

算法思想：若一个样本在特征空间中的 k 个最相似的样本大多数属于某一个类别，则该样本也属于这个类别

 

分类流程：

1. 计算未知样本到每一个训练样本的距离
2. 将训练样本根据距离大小升序排列。
3. 取出距离最近的K个训练样本
4. 进行大多数表决，通过K个样本中哪个类别的样本最多
5. 将未知的样本归宿到出现次数最多的类别

 

回归流程：

1. 计算未知样本到每一个训练样本的距离
2. 将训练样本根据距离大小升序排列。
3. 取出距离最近的 K 个训练样本
4. 把这个 K 个样本的目标值计算其平均值
5. 作为将未知的样本预测的值

 

# api 使用

```
# 1. 导入工具包
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# 2. 特征工程，构建数据
x = [[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
y = [0.1,0.2,0.3,0.4]


# 3. 实例化
# n_neighbors 设置 K值（最近的几个邻居）
model = KNeighborsRegressor(n_neighbors = 3)




# 4. 训练
model.fit(x, y)


# 5.预测
print(model.predict([[4, 4, 5]]))
```

 

# 距离公式

## 欧氏距离

2个点在空间的距离

![img](https://uploader.shimo.im/f/drDRy9Tdfisb9wAM.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

## 曼哈顿距离

![img](https://uploader.shimo.im/f/QrX5IB31VouzFbfL.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

 

## 切比雪夫距离 

![img](https://uploader.shimo.im/f/oS4FCPugSfCLicYz.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

## 闵氏距离

![img](https://uploader.shimo.im/f/Dr92pqeQzJF9rMgw.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

 

# 特征预处理

特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些模型（算法）无法学习到其它的特征。

 

## 归一化

通过对原始数据进行变换把数据映射达到【min, max】(通常是 0-1)之间

 

 

![img](https://uploader.shimo.im/f/3oXb61jsP4JyZmtu.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

 

归一化受到最大值与最小值的影响，这种方法容易受到异常数据的影响, 鲁棒性较差，适合传统精确小数据场景

 

```
from sklearn.preprocessing import MinMaxScaler;


data = [
    [90, 2, 10, 40],
    [60, 4, 15, 45],
    [75, 3, 13, 46]
]
# 初始化归一化对象
# feature_range=(0, 1),默认为0-1
transformer = MinMaxScaler()


print(transformer.fit_transform(data))
```

 

## 数据标准化

通过对原始数据进行标准化，转换为 **均值为0 标准差为**1 的标准正态分布的数据

![img](https://uploader.shimo.im/f/hCk7ns2RWPN97wJ1.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

 

对于标准化来说，如果出现异常点，由于具有一定数据量，少量的异常点对于平均值的影响并不大。

 

正态分布是一种概率分布，大自然很多数据符合正态分布。

方差：是在概率论和统计方差衡量一组数据时离散程度的度量（               其中M为均值  n为数据总数）

![img](https://uploader.shimo.im/f/loGvllUqp7oKGhPa.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

标准差σ是方差开根号：

![img](https://uploader.shimo.im/f/KzDaizDbBPlsd1C9.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

 

 

```
from sklearn.preprocessing import StandardScaler


data = [
    [90, 2, 10, 40],
    [60, 4, 15, 45],
    [75, 3, 13, 46]
]
# 初始化归一化对象
transformer = StandardScaler()


print(transformer.fit_transform(data))


print("mean", transformer.mean_)
print("var", transformer.var_)
```

 

# 代码实战

```
# 导入工具包
from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率


def dm_load_iris():
    iris_data = load_iris()
    #print(iris_data)
    #print(iris_data.keys)
    
    # 查看前5行数据
    print(iris_data.data[:5])
    
    # 查看目标值
    print(iris_data.target)
    
    # 查看特征名称，其实就是列名称
    print(iris_data.feature_names)
    
    # 查看数据集的描述信息.
    print(iris_data.DESCR)
    
def show_iris(): 
    iris_data = load_iris()
    
    # 数据转换为data frame，并且设置列名称
    df = pd.DataFrame(iris_data.data, columns= iris_data.feature_names)
    # 将特征列加入df
    df['label'] =  iris_data.target    
    print(df.head())
    
    # 可视化, x=花瓣长度, y=花瓣宽度, data=iris的df对象, hue=颜色区分, fit_reg=False 不绘制拟合回归线.
    sns.lmplot(x='petal length (cm)', y='petal width (cm)', data=df, hue='label', fit_reg=False)
    plt.title('iris data')
    plt.show()
    
def dm_model():
    # 1 加载数据集
    iris_data = load_iris()
    
    # 2. 划分数据集
    x_train, x_test, y_train, y_test =  train_test_split(
        iris_data.data, # 特征
        iris_data.target, # 标签
        test_size=0.2, 
        random_state=22
    )
    
    
    # 3. 数据预处理（标准化）
    transfer = StandardScaler()
    # fit_transform(): 适用于首次对数据进行标准化处理的情况，通常用于训练集, 能同时完成 fit() 和 transform()。
    # 内部会保存计算出的均值和标准差（供后续使用）
    x_train = transfer.fit_transform(x_train)
    
    # transform(): 适用于对测试集进行标准化处理的情况，通常用于测试集或新的数据. 不需要重新计算统计量。
    #必须先用fit()或fit_transform()计算过均值和标准差
    # 测试集必须使用训练集的均值和标准差，才能保证数据分布一致（避免数据泄漏）
    x_test = transfer.transform(x_test)
    
    # 4. 模型训练
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)
    
    # 5. 模型评估。使用测试集评估
    y_predict = model.predict(x_test)
    print(f"预测结果：{y_predict}")
    
    # 6. 对新数据预测
    my_data = [[5.1, 3.5, 1.4, 0.2]]
    my_data = transfer.transform(my_data)
    my_predict = model.predict(my_data)
    print(f"预测结果：{my_predict}")
    
    # 7. 模型预测概率， 返回每个类别的预测概率
    # 预测概率：[[1. 0. 0.]]
    my_predict_proba = model.predict_proba(my_data)
    print(f"预测概率：{my_predict_proba}")
    
    # 8. 模型预估 直接计算准确率, 100个样本中模型预测正确的个数.
    # 内部自动调用predict(x_test)生成预测值，然后比较预测值与y_test的真实值
    # return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    my_score = model.score(x_test, y_test)
    print(f"my_score：{my_score}")
    
    # 9.模型预估： 方式2: 采用预测值和真实值进行对比, 得到准确率.
    # 需要显式传入预测值y_predict（需先调用model.predict()）
    my_score1 = accuracy_score(y_test, y_predict)
    print(f"my_score1: {my_score1}")
          
          
          
if __name__ == '__main__': 
    dm_model()
```

 

# 超参数选择方法

超参数（Hyperparameters） 是机器学习模型在训练之前需要预先设定的参数，它们不是通过训练数据学习得到的，而是由开发者手动配置或通过自动化搜索确定的。

比如KNN中的K值（邻居个数）就是超参数。

## 交叉验证

一种数据分隔方法，将训练集分成n分，其中一份做验证集（测试集），其他n-1分做训练集。

 

举例：交叉验证法原理：将数据集划分为 cv=4 份

1. 第一次：把第一份数据做验证集，其他数据做训练
2. 第二次：把第二份数据做验证集，其他数据做训练
3. 以此类推，总共训练4次，评估4次。

使用训练集+验证集多次评估模型，取平均值做交叉验证为模型得分

若k=5模型得分最好，再使用全部训练集(训练集+验证集) 对k=5模型再训练一边，再使用测试集对k=5模型做评估

 

交叉验证法，是划分数据集的一种方法，目的就是为了得到更加准确可信的模型评分。

 

最优参数下交叉验证的平均表现：对每个超参数组合（比如knn中n_neighbors=3），计算k次验证得分的平均值（如准确率的平均值）。

![img](https://uploader.shimo.im/f/tY3zeZ4iixjFDdMt.png!thumbnail?accessToken=eyJhbGciOiJIUzI1NiIsImtpZCI6ImRlZmF1bHQiLCJ0eXAiOiJKV1QifQ.eyJleHAiOjE3NTk4MDQwOTgsImZpbGVHVUlEIjoiOTEzSk1KRGwyZVMweWJBRSIsImlhdCI6MTc1OTgwMzc5OCwiaXNzIjoidXBsb2FkZXJfYWNjZXNzX3Jlc291cmNlIiwicGFhIjoiYWxsOmFsbDoiLCJ1c2VySWQiOjk3MzY0OTQ5fQ.n1UtsPIq0dehAYQuL4Q-HF7y4Tbuced_S1p4S-n7qQU)

 

 

## 网格搜索

模型有很多超参数，其能力存在很大差异。需要手动产生很多超参数组合，来训练模型。

每组超参数都采用交叉验证评估，最后选出最优参数组合建立模型。

 

网格搜索是模型调参的有力工具。寻找最优超参数的工具！

只需要将若干参数传递给网格搜索对象，它自动帮我们完成不同超参数的组合、模型训练、模型评估，最终返回一组最优的超参数。

 

网格搜索 + 交叉验证的强力组合 (模型选择和调优)：

1. 交叉验证解决模型的数据输入问题(数据集划分)得到更可靠的模型
2. 网格搜索解决超参数的组合
3. 两个组合再一起形成一个模型参数调优的解决方案

 

## 代码

```
from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
from sklearn.model_selection import train_test_split, GridSearchCV    # 分割训练集和测试集的, 网格搜索的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率


# 1. 加载数据集
iris_data = load_iris()


# 2. 划分数据集
x_train, x_test, y_train, y_test = train_test_split(
    iris_data.data, 
    iris_data.target,
    test_size=0.2, 
    random_state=22
)


# 3。 数据预处理，标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)


# 4. 创建评估器对象
estimator = KNeighborsClassifier()


# 5.使用校验验证网格搜索
param_grid = {"n_neighbors": range(1, 10)}


# 6. 网格搜索过程 + 交叉验证
# estimator：定义模型
# param_grid: 定义knn模型超数，表示“使用多少个最近邻样本进行预测， 
# range(1, 10)表示测试n_neighbors从1到9的所有整数值（共9种情况）
# cv：交叉验证，这里将数据分成5份，轮流用4份训练，1份验证
estimator = GridSearchCV(
    estimator = estimator,
    param_grid=param_grid,
    cv = 5
)


# 7. 训练过程
estimator.fit(x_train, y_train)


# 8. 擦好看结果
# best_score_：模型在交叉验证中取得的最高平均分数。就是在最优参数下交叉验证的平均表示
# best_estimator_：最优超参数对应的模型对象，可直接用此模型进行预测
# best_params_：返回最优超参数的组合，字典形式
# cv_results_： 返回所有超参数组合的详细结果（字典形式，包含训练/验证分数、时间等）
print(f'''
best_score_={estimator.best_score_}
best_estimator_={estimator.best_estimator_}
cv_results_={estimator.cv_results_}
best_params_={estimator.best_params_}
''')


# 9. 得到最优模型后，对模型重新预
# 使用best_params_的最优超参数
new_estimator = KNeighborsClassifier(n_neighbors=6)
new_estimator.fit(x_train, y_train)
# 因为数据量和特征的问题, 该值可能小于上述的平均测试得分
print(f'模型评估: {estimator.score(x_test, y_test)}')
```

 

 

 

 