#导入所需的类和函数
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



#得到相同的随机数序列,这有助于我们调试
seed = 7
numpy.random.seed(seed)


#使用pandas库加载数据集
dataframe = pandas.read_csv("data/sonar.csv", header=None) # 加载数据集
dataset = dataframe.values
X = dataset[:, 0:60].astype(float) # 分割为60个输入变量
Y = dataset[:, 60] # 1个输出变量

#将输出的字符串类型转换为0，1类型
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

#调用models模型
model=create_model1()
#随机打乱顺序
indices = numpy.random.permutation(X.shape[0]) # shape[0]表示第0轴的长度，通常是训练数据的数量
rand_data_x = X[indices]
rand_data_y = encoded_Y[indices] # encoded_Y就是标记（label）

# Fit the model
history = model.fit(rand_data_x, rand_data_y, validation_split=0.5,epochs=300, batch_size=16, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

	
# #evaluate baseline model with standardized dataset
# start_time = time.time()
# numpy.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=create_model1, epochs=300, batch_size=16, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
# print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# end_time = time.time()
# print("time consumed: ", end_time - start_time)