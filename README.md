# sonar

1.1 数据集描述
这是一个从多个服务收集回来描述声纳返回反射的数据集。60个输入变量描述了在不同角度的反射强度。这是一个二元分类问题,需要一个模型来区分岩石和金属圆筒。
这是一个很好理解的数据集。所有的变量是连续的,一般在0到1的范围。输出变量是一个字符串“M”和“R”我岩石,它将需要转化为整数 1 和 0 。
使用这个数据集的一个好处是,它是一个标准的基准问题。这意味着我们有一个好的模型的预期能力的想法。使用交叉验证,神经网络能够实现性能大约84%的上限为自定义模型精度在88%左右。

1.2 参数设置及说明
1.导入所需的类和函数
import time
import numpy
import pandas
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from models import create_baseline,create_smaller,create_larger,create_model1,create_model2,create_model3

注：models是自定义的model函数集合
2.得到相同的随机数序列,这有助于我们调试
seed = 7
numpy.random.seed(seed)

3.使用pandas库加载数据集
dataframe = pandas.read_csv("data/sonar.csv", header=None) # 加载数据集
dataset = dataframe.values
X = dataset[:, 0:60].astype(float) # 分割为60个输入变量
Y = dataset[:, 60] # 1个输出变量


4．将输出的字符串类型转换为0，1类型
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y) 

5. 随机打乱顺序
indices = numpy.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量
rand_data_x = X[indices]
rand_data_y = encoded_Y[indices] # encoded_Y就是标记（label）

注1:
shuffle和validation_split的顺序
模型的fit函数有两个参数，shuffle用于将数据打乱，validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集
这里有个陷阱是，程序是先执行validation_split，再执行shuffle的，所以会出现这种情况：
假如你的训练集是有序的，比方说正样本在前负样本在后，又设置了validation_split，那么你的验证集中很可能将全部是负样本
同样的，这个东西不会有任何错误报出来，因为Keras不可能知道你的数据有没有经过shuffle，保险起见如果你的数据是没shuffle过的，最好手动shuffle一下

注2：
Sonar数据集是有序的，手动shuffle数据
1.3 模型设计
1.3.1 baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
return model

1.3.2 smaller model
def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=60, kernel_initializer ='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer ='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
return model

1.3.3 larger model
def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
      # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
1.4 模型dropout优化
1.4.1对输入层使用Dropout正则化
def create_model1():
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    return model
		
1.4.2对隐层使用Dropout正则化
def create_model2():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))) 
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)  
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    return model
	
1.4.3对输入和隐层同时正则化
def create_model3():
    # create model
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3))) 
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)  
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
return model
1.5评估实验模型
start_time = time.time()
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model3, epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
end_time = time.time()
print("time consumed: ", end_time - start_time)
1.6 可视化Keras的模型训练历史&实验结果分析
  

注:使用模型create_model1(Input layers dropout优化模型)

标准Model：
	Baseline	Small	Large
Standardized(mean,std)	85.04% (7.02%)	83.18% (8.04%)	4.56% (7.28%)
time consumed	42.81956696510315	43.14090275764465	47.636380434036255

Dropout 优化Model
	Input layers_drop	Hidden layers_drop	Input&Hidden layers_drop
Standardized(mean,std)	86.52% (5.64%)	84.54% (7.21%)	85.06% (6.41%)
time consumed	50.24324107170105	51.52637267112732	51.96850395202637

注1：声纳数据集：https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/

注2：
epochs=300, batch_size=16

实验环境：win10 X64 i5-4200H 2.8GHz  python3.5
