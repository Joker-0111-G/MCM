#!/usr/bin/env python
# coding: utf-8

# In[331]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree 
import graphviz 
from sklearn import preprocessing
#%matplotlib
from sklearn.neural_network import MLPRegressor


# In[332]:


# #np.polyfit使用例程
# x = np.linspace(start=0.4,stop=0.15,num=1000)
# #曲线A
# z1 = np.polyfit(df_loss.values[:,0], df_loss.values[:,1], 3)
# p1 = np.poly1d(z1) #使用次数合成多项式
# y_pre = p1(x)
# plt.plot(x,y_pre)

#导入表一数据并进行预处理
# df1 = pd.read_excel("1.xlsx")
# df1 = df1.dropna()#抛弃丢失值
# df1 = df1.iloc[:,1:]
# df1 = pd.get_dummies(df1)#独热编码，并且删除倒数第二列，即label为1时表示有风化，label为0时表示无风化
# temp = df1.iloc[:,-1]
# df1 = df1.iloc[:,:-2]
# df1 = df1.join(temp)


# In[333]:


df_impr = pd.read_excel("2_1_new.xlsx")
xtemp = pd.get_dummies(df_impr.iloc[:,-4:])
ytemp = df_impr.iloc[:,1:-4]
#xtemp = sm.add_constant(xtemp)
model = sm.OLS(ytemp.iloc[:,11],xtemp)
result = model.fit()
result.summary()


# In[334]:


#第一题第三问多元线性回归
df_train = pd.read_excel("train5.xlsx")
df_train = df_train.dropna()
x_train = pd.get_dummies(df_train.iloc[:,-4:])
x_train = x_train.iloc[:,1:]
x_train = x_train.drop(labels=["颜色_绿"],axis=1)
y_train = df_train.iloc[:,1:-5]


# In[335]:


y_train
x_train


# In[336]:


test = x_train.iloc[:,4:]
y_train.shape
x_train.shape


# In[337]:


#2022年9月18日10:04:02基于神经网络对预测效果的优化

#x_train = sm.add_constant(x_train)

#model = sm.GLS(y_train.iloc[:,0],x_train)
#result = model.fit()
#result.summary()
df_train = pd.read_excel("train5.xlsx")
#df_train = df_train.dropna()
x_train = pd.get_dummies(df_train.iloc[:,-4:])
x_train = x_train.iloc[:,1:]
x_train = x_train.drop(labels=["颜色_绿"],axis=1)
y_train = df_train.iloc[:,1:-5]
for i in range(5):
    model_mlp = MLPRegressor(
        hidden_layer_sizes=(18,2),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False,  nesterovs_momentum=True,
        early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08,momentum=0.9)#momentum=0.9,

    model_mlp.fit(x_train,y_train.iloc[:,i])
    pred = model_mlp.predict(x_train)
    #x_train = x_train.reshape(-1,1)
    mlp_score=model_mlp.score(x_train.values,y_train.iloc[:,i].values)
    mlp_score
    draw, = plt.plot(y_train.iloc[:,i], 'b')
    draw.set_label("true")
    draw, = plt.plot(pred, 'r')
    draw.set_label("predict")
    #plt.plot(x1, result, 'ro')
    draw.set_label("predict")
    plt.ylabel('P2O5')
    plt.legend()
    plt.show()
    print(mlp_score)


# In[338]:


df_pred.iloc[:,1:]


# In[339]:


df_pred = pd.read_excel("test_test.xlsx")
df_pred.loc[:,['二氧化硅(SiO2)','氧化钙(CaO)','氧化铝(Al2O3)','氧化铜(CuO)','氧化钡(BaO)','五氧化二磷(P2O5)']]


# In[342]:


# w = np.array(result.params)
# pred = x_train.values*w
# pred = pred.sum(axis=1)
# x = np.arange(1,17)
# plt.plot(x,pred,'b')
# plt.plot(x,y_train.iloc[:,0].values,'r')
# plt.show()


# In[344]:


#进行风化样本的预测
df_pred = pd.read_excel("test_test.xlsx")


# In[345]:


df_pred = pd.read_excel("test_test.xlsx")
df_pred = df_pred.dropna()
x_pred = pd.get_dummies(df_pred.iloc[:,-4:])
y_pred = df_train.iloc[:,1:-5]


# In[346]:


x_pred = x_pred.iloc[:,1:]
x_pred


# In[347]:


x_pred = x_pred.drop(labels=["纹饰_B"],axis = 1)


# In[348]:


w


# In[349]:


# y_pred = x_pred.values*w


# In[350]:


# y_pred = y_pred.sum(axis=1)


# In[351]:


# bias = np.random.normal(loc=0.0, scale=1.0, size=[y_pred.shape[0]])


# In[352]:


# y_pred = y_pred + bias
# y_pred


# In[353]:


Outer = pd.DataFrame(y_pred,columns=["预测"])
Outer.to_excel(excel_writer=r"2022数据处理2.xlsx")


# In[354]:


#五氧化二磷
df_train = pd.read_excel("train6.xlsx")
df_train = df_train.dropna()
x_train = pd.get_dummies(df_train.iloc[:,-4:])
x_train = x_train.iloc[:,1:]
y_train = df_train.iloc[:,1:-5]
x_train = x_train.drop(labels=["颜色_绿"],axis=1)
#x_train = sm.add_constant(x_train)



model = sm.GLS(y_train.iloc[:,10],x_train)
result = model.fit()
result.summary()


# In[355]:


#画图
w = np.array(result.params)
pred = x_train.values*w
pred = pred.sum(axis=1)
x = np.arange(1,26)
plt.plot(x,pred,'b')
plt.plot(x,y_train.iloc[:,10].values,'r')
plt.show()


# In[356]:


#预测
y_pred = x_pred.values*w
y_pred = y_pred.sum(axis=1)
bias = np.random.normal(loc=0.0, scale=1.0, size=[y_pred.shape[0]])
y_pred = y_pred + bias
y_pred


# In[357]:


x_pred


# In[358]:


#导出数据
Outer = pd.DataFrame(y_pred,columns=["预测五氧化二磷"])
Outer.to_excel(excel_writer=r"2022数据处理3.xlsx")


# In[359]:


#氧化钙
df_train = pd.read_excel("train6.xlsx")
df_train = df_train.dropna()
x_train = pd.get_dummies(df_train.iloc[:,-4:])
x_train = x_train.iloc[:,1:]
x_train = x_train.drop(labels=["颜色_绿"],axis=1)
y_train = df_train.iloc[:,1:-5]
x_train = sm.add_constant(x_train)
model = sm.GLS(y_train.iloc[:,5],x_train)
result = model.fit()
result.summary()


# num=5#氧化
# model_mlp = MLPRegressor(
#     hidden_layer_sizes=(18,2),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
#     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
#     random_state=1, tol=0.0001, verbose=False, warm_start=False,  nesterovs_momentum=True,
#     early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08,momentum=0.9)#momentum=0.9,

# model_mlp.fit(x_train,y_train.iloc[:,num])
# pred = model_mlp.predict(x_train)
# #x_train = x_train.reshape(-1,1)
# mlp_score=model_mlp.score(x_train.values,y_train.iloc[:,num].values)
# mlp_score
# plt.plot(y_train.iloc[:,num], 'b')
# plt.plot(pred, 'r')
# #plt.plot(x1, result, 'ro')
# plt.show()
# print(mlp_score)


# In[360]:


#画图
w = np.array(result.params)
pred = x_train.values*w
pred = pred.sum(axis=1)
x = np.arange(1,26)
draw, = plt.plot(x,pred,'b')
draw.set_label("pred")
draw, =  plt.plot(x,y_train.iloc[:,10].values,'r')
draw.set_label("true")
plt.ylabel("score of ols")
plt.legend()
plt.show()


# In[363]:


#预测
y_pred = x_pred.values*w[:-1]
y_pred = y_pred.sum(axis=1)
bias = np.random.normal(loc=0.0, scale=1.0, size=[y_pred.shape[0]])
y_pred = y_pred + bias
y_pred


# In[364]:


#导出数据
Outer = pd.DataFrame(y_pred,columns=["预测氧化钙"])
Outer.to_excel(excel_writer=r"2022数据处理4.xlsx")


# In[365]:


#预测氧化铜
df_train = pd.read_excel("train7.xlsx")
df_train = df_train.dropna()
x_train = pd.get_dummies(df_train.iloc[:,-4:])
x_train = x_train.iloc[:,1:]
x_train = x_train.drop(labels=["颜色_绿"],axis=1)
y_train = df_train.iloc[:,1:-5]
model = sm.GLS(y_train.iloc[:,7],x_train)
result = model.fit()
print(result.summary())

num=8#氧化铅
model_mlp = MLPRegressor(
    hidden_layer_sizes=(18,2),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False,  nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08,momentum=0.9)#momentum=0.9,

model_mlp.fit(x_train,y_train.iloc[:,num])
pred = model_mlp.predict(x_train)
#x_train = x_train.reshape(-1,1)
mlp_score=model_mlp.score(x_train.values,y_train.iloc[:,num].values)
mlp_score
plt.plot(y_train.iloc[:,num], 'b')
plt.plot(pred, 'r')
#plt.plot(x1, result, 'ro')
plt.show()
print(mlp_score)


# In[366]:


#预测氧化铝
df_train = pd.read_excel("train7.xlsx")
df_train = df_train.dropna()
x_train = pd.get_dummies(df_train.iloc[:,-4:])
x_train = x_train.iloc[:,1:]
x_train = x_train.drop(labels=["颜色_绿"],axis=1)
y_train = df_train.iloc[:,1:-5]
model = sm.GLS(y_train.iloc[:,5],x_train)
result = model.fit()
print(result.summary())

num=5#氧化吕
model_mlp = MLPRegressor(
    hidden_layer_sizes=(18,2),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False,  nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08,momentum=0.9)#momentum=0.9,

model_mlp.fit(x_train,y_train.iloc[:,num])
pred = model_mlp.predict(x_train)
#x_train = x_train.reshape(-1,1)
mlp_score=model_mlp.score(x_train.values,y_train.iloc[:,num].values)
mlp_score
plt.plot(y_train.iloc[:,num], 'b')
plt.plot(pred, 'r')
#plt.plot(x1, result, 'ro')
plt.show()
print(mlp_score)


# In[369]:


#预测
y_pred = x_pred.values*w[:-1]
y_pred = y_pred.sum(axis=1)
bias = np.random.normal(loc=0.0, scale=1.0, size=[y_pred.shape[0]])
y_pred = y_pred + bias
y_pred


# In[370]:


#导出数据
Outer = pd.DataFrame(y_pred,columns=["预测氧化铜"])
Outer.to_excel(excel_writer=r"2022数据处理5.xlsx")


# In[371]:


#二氧化硫
df_train = pd.read_excel("train8.xlsx")
df_train = df_train.dropna()
x_train = pd.get_dummies(df_train.iloc[:,-4:])
x_train = x_train.iloc[:,1:]
x_train = x_train.drop(labels=["颜色_绿"],axis=1)
y_train = df_train.iloc[:,1:-5]
model = sm.GLS(y_train.iloc[:,9],x_train)
result = model.fit()
print(result.summary())


num=9#氧化呗
model_mlp = MLPRegressor(
    hidden_layer_sizes=(18,2),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False,  nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08,momentum=0.9)#momentum=0.9,

model_mlp.fit(x_train,y_train.iloc[:,num])
pred = model_mlp.predict(x_train)
#x_train = x_train.reshape(-1,1)
mlp_score=model_mlp.score(x_train.values,y_train.iloc[:,num].values)
mlp_score
plt.plot(y_train.iloc[:,num], 'b')
plt.plot(pred, 'r')
#plt.plot(x1, result, 'ro')
plt.show()
print(mlp_score)


# In[372]:


#画图
w = np.array(result.params)
pred = x_train.values*w
pred = pred.sum(axis=1)
x = np.arange(1,26)
plt.plot(x,pred,'b')
plt.plot(x,y_train.iloc[:,10].values,'r')
plt.show()


# In[373]:


#预测
y_pred = x_pred.values*w
y_pred = y_pred.sum(axis=1)
bias = np.random.normal(loc=0.0, scale=1.0, size=[y_pred.shape[0]])
y_pred = y_pred + bias
y_pred


# In[374]:


#导出数据
Outer = pd.DataFrame(y_pred,columns=["预测二氧化硫"])
Outer.to_excel(excel_writer=r"2022数据处理6.xlsx")


# In[375]:


#读取数据
#风化：1 无风化：0
#高钾：1 铅钡：0     
df_lr = pd.read_excel("2_1_new.xlsx")
df_lr


# In[376]:


x_train_lr = df_lr.iloc[:,1:-4]
y_train_lr = df_lr.iloc[:,-3]
x_train,x_test,y_train,y_test = train_test_split(x_train_lr,y_train_lr,test_size=0.2)


# In[377]:


#x_train = sm.add_constant(x_train)
model = sm.OLS(y_train,x_train)
result = model.fit()
result.summary()


# In[378]:


#Topsis函数
#Topsis函数定义
#0 极大型 
#1 极小型
#2 中间型 额外标志：Xbest #PS:个人感觉，中间型的转换，本质上是先转换成极小型再转换成极大型
#3 区间型 额外标志：[a,b]区间 #正向化的方法是不唯一的，这里参考网课上的
#正向化
def ForwardDirection(m,y):#m表示输入矩阵，y表示指标类型
    num=len(y)
    for i in range(num):
        #i代表着第i个指标，对应着m的第i列（从0计数）
        labeltype=y[i]
        if(labeltype[0]==0):
            print("第",i+1,"个指标为极大型指标，无需正向化")
        elif(labeltype[0]==1):
            print("第",i+1,"个指标为极小型指标，进行正向化")
            maxnum=np.amax(m,axis=0)#返回每一列元素的最大值
            maxnum=maxnum[i]
            m[:,i]= maxnum-m[:,i]
        elif(labeltype[0]==2):
            print("第",i+1,"个指标为中间型指标，进行正向化")
            xbest=labeltype[1]
            M=np.max(np.abs(m[:,i]-xbest))
            m[:,i]=(1-np.abs(m[:,i]-xbest)/M)
        elif(labeltype[0]==3):
            print("第",i+1,"个指标为区间型指标，进行正向化")
            a=labeltype[1]
            b=labeltype[2]
           # print("a,b:",a,b)
            minx=np.min(m[:,i])
            maxx=np.max(m[:,i])
            #print(m[:,i])
           # print("min,max",minx,maxx)
            M=max(a-minx,maxx-b)
            col=len(m[:,i])
            for j in range(col):
                if(m[j,i]<a):
                    m[j,i]=1-(a-m[j,i])/M
                elif(m[j,i]<=b and m[j,i]>=a):
                    m[j,i]=1
                elif(m[j,i]>b):
                    m[j,i]=1-(m[j,i]-b)/M
    return m

def Standardize(m):#m为输入矩阵，且已经正向化
    
    #第一步，先把m中元素乘方
    temp_all=np.power(m,2)

    #第二步，按列求和得到一个行向量
    temp_all=np.sum(temp_all,axis=0)#在第一个轴展开方向上求和

    #第三步，将行向量元素开方
    temp_all=np.power(temp_all,1/2)

    #第四步，将m中每一列元素除以第三部中行向量对应列元素
    for i in range(len(temp_all)):
        m[:,i]=m[:,i]/temp_all[i]
       
    return m

#计算得分
def getGrade(m,w):#m为输入矩阵，且已经标准化
    
    #第一步，得到每一列的最大值Zmax与最小值Zmin
    Zmax=np.max(m,axis=0)
    Zmin=np.min(m,axis=0)
    
    #第二步，通过ZmaxZmin得到每个样本（即本例中的小王小明等）的每个指标到各个指标最大值的距离与到最小值的距离
        #1.将m中的每一列元素减去Zmax/Zmin中对应列的值
        #2.然后将结果乘方
        #3.按列方向展开求和
        #4.再开方，从而得到每个评价对象（即样本）的Dmax与Dmin（注意：这里的算法我还没有算上权重）
    Dmax = np.subtract(m,Zmax)#Zmax是行向量，这里有广播
   # print(Dmax)
    Dmin = np.subtract(m,Zmin)#同上
    Dmax=np.power(Dmax,2)
    Dmin=np.power(Dmin,2)
    
    Dmax=Dmax*w
    Dmin=Dmin*w
    
    Dmax=np.sum(Dmax,axis=1)
    Dmin=np.sum(Dmin,axis=1)
 
    Dmax=np.power(Dmax,1/2)
    Dmin=np.power(Dmin,1/2)
    #print("Dmax,Dmin",Dmax,"\n",Dmin)
    #第三步，根据公式：分数=到最小值距离/(到最大值距离+到最小值距离)，算出每个评价对象的综合得分S
    S=Dmin/(Dmax+Dmin)
    #print(S)
    #第四步，对得分进行归一化
    S_sum=np.sum(S,axis=0)
    S=S/S_sum
    return S


# In[379]:


#读入铅钡数据
df_topsis_qb = pd.read_excel("铅钡Topsis.xlsx")


# In[380]:


df_topsis_qb


# In[381]:


#Topsis成分：二氧化硅 氧化钙 氧化铜 五氧化二磷 二氧化硫
data = df_topsis_qb.loc[:,['二氧化硅(SiO2)','氧化钙(CaO)','氧化铜(CuO)','氧化铅(PbO)','五氧化二磷(P2O5)','二氧化硫(SO2)']]
#考虑到原始数据具有较多0值，故对权重进行处理
w=[[1/6,1/6,1/6,1/6,1/6,1/6]]*data.shape[0]
w=np.array(w)
data=np.array(data,dtype=np.float32)
label=[[1],[0],[1],[0],[0],[0]]
data=ForwardDirection(data,label)
data=Standardize(data)
S=getGrade(data,w)


# In[382]:


S_df = pd.DataFrame(S,columns=['Topsis得分'])
S_df.to_excel(excel_writer=r"2022C数据处理Topsis1.xlsx")


# In[383]:


#读入高钾数据
df_topsis_gj = pd.read_excel("高钾Topsis.xlsx")


# In[384]:


data = df_topsis_gj.loc[:,['二氧化硅(SiO2)','氧化钾(K2O)','氧化钙(CaO)','氧化铝(Al2O3)','五氧化二磷(P2O5)']]
#考虑到原始数据具有较多0值，故对权重进行处理
w=[[1/5,1/5,1/5,1/5,1/5]]*data.shape[0]
w=np.array(w)
data=np.array(data,dtype=np.float32)
label=[[0],[1],[1],[1],[1]]
data=ForwardDirection(data,label)
data=Standardize(data)
S=getGrade(data,w)


# In[385]:


S_df = pd.DataFrame(S,columns=['Topsis得分'])
S_df.to_excel(excel_writer=r"2022C数据处理Topsis0.xlsx")


# In[386]:


#用决策树进行种类预测
df_tree = pd.read_excel("决策树预测.xlsx")
df_tree = df_tree.iloc[:,1:]
x_train,x_test,y_train,y_test = train_test_split(df_tree.iloc[:,:-1],df_tree.iloc[:,-1],test_size=0.3)
#生成模型
clf = tree.DecisionTreeClassifier(criterion="entropy",
                                  random_state=30,
                                  max_depth=5)

clf = clf.fit(x_train,y_train)
result = clf.score(x_test,y_test) 
feature_name = ['二氧化硅(SiO2)','氧化钠(Na2O)','氧化钾(K2O)','氧化钙(CaO)','氧化镁(MgO)','氧化铝(Al2O3)',
                '氧化铁(Fe2O3)','氧化铜(CuO)','氧化铅(PbO)','氧化钡(BaO)','五氧化二磷(P2O5)','氧化锶(SrO)',
                '氧化锡(SnO2)','二氧化硫(SO2)','表面风化']
tree.plot_tree(clf,class_names=['高钾','铅钡'])


# In[387]:


x_train


# In[388]:


[*zip(feature_name,clf.feature_importances_)] 


# In[389]:


clf.score(x_test,y_test),clf.score(x_train,y_train)


# In[390]:


#针对树深度的灵敏度分析
x_train,x_test,y_train,y_test = train_test_split(df_tree.iloc[:,:-1],df_tree.iloc[:,-1],test_size=0.3)
score_train = []
score_test = []
for depth in range(1,10):
    clf = tree.DecisionTreeClassifier(criterion="entropy",
                                  random_state=30,
                                  max_depth=depth,
                                  splitter="random")

    clf = clf.fit(x_train,y_train)
    score_train.append(clf.score(x_train,y_train))
    score_test.append(clf.score(x_test,y_test))
score_train,score_test


# In[391]:


x = [1,2,3,4,5,6,7,8,9]
draw, = plt.plot(x,score_train,'b')
draw.set_label("train")
draw, = plt.plot(x,score_test,'r')
draw.set_label("test")
plt.xlabel('max_depth')
plt.legend()


# In[392]:


score_train=[]
score_test = []
#针对数据划分的灵敏度分析
for i in np.linspace(0.1,0.9,9):
    x_train,x_test,y_train,y_test = train_test_split(df_tree.iloc[:,:-1],df_tree.iloc[:,-1],test_size=i)
    clf = tree.DecisionTreeClassifier(criterion="entropy",
                                  random_state=30,
                                  max_depth=5,
                                  splitter="random")

    clf = clf.fit(x_train,y_train)
    score_train.append(clf.score(x_train,y_train))
    score_test.append(clf.score(x_test,y_test))
x = [1,2,3,4,5,6,7,8,9]
plt.plot(x,score_train,'b')
plt.plot(x,score_test,'r')


# In[393]:


#   ,min_samples_leaf=10,min_samples_split=10  
#针对min_samples_leaf参数进行敏感性分析
x_train,x_test,y_train,y_test = train_test_split(df_tree.iloc[:,:-1],df_tree.iloc[:,-1],test_size=0.3)
score_train=[]
score_test = []
for i in range(20):
    clf = tree.DecisionTreeClassifier(criterion="entropy",
                              random_state=30,
                              max_depth=4,
                              splitter="random",
                                min_samples_leaf=i+1)

    clf = clf.fit(x_train,y_train)
    score_train.append(clf.score(x_train,y_train))
    score_test.append(clf.score(x_test,y_test))
x = np.arange(20)
draw, = plt.plot(x,score_train,'b')
draw.set_label("train")
draw, = plt.plot(x,score_test,'r')
draw.set_label("test")
plt.xlabel('min_samples_leaf')
plt.legend()


# In[394]:


#针对min_samples_spilt参数进行敏感性分析
x_train,x_test,y_train,y_test = train_test_split(df_tree.iloc[:,:-1],df_tree.iloc[:,-1],test_size=0.3)
score_train=[]
score_test = []
for i in range(20):
    clf = tree.DecisionTreeClassifier(criterion="entropy",
                              random_state=30,
                              max_depth=4,
                              splitter="random",
                                min_samples_split=i+2)

    clf = clf.fit(x_train,y_train)
    score_train.append(clf.score(x_train,y_train))
    score_test.append(clf.score(x_test,y_test))
x = np.arange(20)
draw, = plt.plot(x,score_train,'b')
draw.set_label("train")
draw, = plt.plot(x,score_test,'r')
draw.set_label("test")
plt.xlabel('min_samples_spilt')
plt.legend()


# In[395]:


#确定参数maxdp = 5
df_tree = pd.read_excel("决策树预测.xlsx")
df_tree = df_tree.iloc[:,1:]
x_train,x_test,y_train,y_test = train_test_split(df_tree.iloc[:,:-1],df_tree.iloc[:,-1],test_size=0.3)
#生成模型
clf = tree.DecisionTreeClassifier(criterion="entropy",
                                  random_state=30,
                                  max_depth=5)

clf = clf.fit(x_train,y_train)
result = clf.score(x_test,y_test) 
feature_name = ['二氧化硅(SiO2)','氧化钠(Na2O)','氧化钾(K2O)','氧化钙(CaO)','氧化镁(MgO)','氧化铝(Al2O3)',
                '氧化铁(Fe2O3)','氧化铜(CuO)','氧化铅(PbO)','氧化钡(BaO)','五氧化二磷(P2O5)','氧化锶(SrO)',
                '氧化锡(SnO2)','二氧化硫(SO2)','表面风化']
tree.plot_tree(clf,class_names=['1','0'])#1:高钾 0：铅钡
#clf.score(x_train,y_train),clf.score(x_test,y_test)


# In[396]:


#导入预测数据
df_pred_tree = pd.read_excel("决策树附件3.xlsx")
df_pred_tree


# In[397]:


x=df_pred_tree.iloc[:,1:]


# In[398]:


x


# In[399]:


clf.predict_proba(x.iloc[:,0:-2])


# In[400]:


#第四题
df_4 = pd.read_excel("高钾_4.xlsx")


# In[401]:


df_4


# In[402]:


#消除量纲
df_stand = preprocessing.StandardScaler().fit_transform(df_4.iloc[:,1:])


# In[403]:


x = np.arange(18)
#topsis
plt.scatter(x,df_stand[:,-1])
#sio2
plt.scatter(x,df_stand[:,0])
#plt.scatter(x,df_stand[:,1])
plt.scatter(x,df_stand[:,2])


# In[404]:


#灰色关联手撕
#第一步，数据预处理：每个元素除以均值
#第二步，计算关联系数：
    #1）求每一列特征的两极最小差与两极最大差a与b
    #1.5)求得母与子对应列的绝对差fabs
    #2）定义分辨系数p （超参数，一般为0.5）
    #3）求得子列每个元素的Yi = (a+pb)/(fabs+pb)
    #4）对每一列的Yi求和/n，最终得到每一个特征的关联程度

def gray(df_values,p):

    temp1 = np.sum(df_values,axis=0)#得到每一列的和

    temp1 = temp1/df_values.shape[0]#得到每一列的均值

    df_values = df_values/temp1 #对数据预处理

    #print(df_values)
    df_mother = df_values[:,-1]#母列
    df_son = df_values[:,:-1]#子列
    for i in range(df_son.shape[1]):
        df_son[:,i] = np.fabs(df_son[:,i] - df_mother)
    #print(df_son)
    a=np.min(df_son)#求每一列特征的两极最小差与两极最大差a与b
    b=np.max(df_son)
    #print(a,b)
    df_son = (a+p*b)/(df_son+p*b)
    #print(df_son)
    df_son = np.sum(df_son,axis=0)/df_values.shape[0]
    
    return df_son


# In[405]:


df_4 = pd.read_excel("高钾_4.xlsx")
w = gray(df_4.iloc[:,1:].values,0.5)#高钾
w = pd.DataFrame(w,columns=["灰色关联度"])
w.to_excel(excel_writer=r"灰色关联度_高钾.xlsx")

df_4_qb = pd.read_excel("铅钡_4.xlsx")
w = gray(df_4_qb.iloc[:,1:].values,0.5)#高钾
w = pd.DataFrame(w,columns=["灰色关联度"])
w.to_excel(excel_writer=r"灰色关联度_铅钡.xlsx")


# In[406]:


#灰色关联的敏感性分析_二氧化硅——高钾玻璃
df_4 = pd.read_excel("高钾_4.xlsx")
res = []
x = np.linspace(0.1,2,100)
bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
data = df_4.iloc[:,1:].values
for i in x:
    data = df_4.iloc[:,1:].values
    bias = np.random.random([data.shape[0],data.shape[1]])
    #bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
    #print(bias)
    data = data+bias
   # print(data)
    w=gray(data,0.5)
    SiO2 = w[0]
    res.append(SiO2)
   # print(np.sum(w))
plt.ylim(0,1)
draw, = plt.plot(res,'r')
draw.set_label('SiO2-1')
plt.ylabel("degree of gray")
plt.legend()

res = []
x = np.linspace(0.1,2,100)
bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
data = df_4.iloc[:,1:].values
for i in x:
    data = df_4.iloc[:,1:].values
    bias = np.random.random([data.shape[0],data.shape[1]])
    #bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
    #print(bias)
    data = data+bias
   # print(data)
    w=gray(data,0.5)
    SiO2 = w[2]
    res.append(SiO2)
   # print(np.sum(w))
plt.ylim(0,1)
draw, = plt.plot(res,'b')
draw.set_label('K2O-1')
plt.ylabel("degree of gray")
plt.legend()
#res

df_4 = pd.read_excel("铅钡_4.xlsx")
res = []
x = np.linspace(0.1,2,100)
bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
data = df_4.iloc[:,1:].values
for i in x:
    data = df_4.iloc[:,1:].values
    bias = np.random.random([data.shape[0],data.shape[1]])
    #bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
    #print(bias)
    data = data+bias
   # print(data)
    w=gray(data,0.5)
    SiO2 = w[2]
    res.append(SiO2)
   # print(np.sum(w))
plt.ylim(0,1)
draw, = plt.plot(res,'y')
draw.set_label('SiO2-0')
plt.ylabel("degree of gray")
plt.legend()

res = []
x = np.linspace(0.1,2,100)
bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
data = df_4.iloc[:,1:].values
for i in x:
    data = df_4.iloc[:,1:].values
    bias = np.random.random([data.shape[0],data.shape[1]])
    #bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
    #print(bias)
    data = data+bias
   # print(data)
    w=gray(data,0.5)
    SiO2 = w[2]
    res.append(SiO2)
   # print(np.sum(w))
plt.ylim(0,1)
draw, = plt.plot(res,'g')
draw.set_label('K2O-0')
plt.ylabel("degree of gray")
plt.legend()


# In[407]:


#灰色关联的敏感性分析——氧化钾——高钾玻璃
df_4 = pd.read_excel("铅钡_4.xlsx")
for j in range(14):
    res = []
    x = np.linspace(0.1,2,100)
    bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
    data = df_4.iloc[:,1:].values
    for i in x:
        data = df_4.iloc[:,1:].values
        bias = np.random.random([data.shape[0],data.shape[1]])
        #bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
        #print(bias)
        data = data+bias
       # print(data)
        w=gray(data,0.5)
        SiO2 = w[j]
        res.append(SiO2)
       # print(np.sum(w))
    plt.ylim(0,1)
    draw, = plt.plot(res,'r')
    print(df_4.iloc[:1:].columns[0])
    draw.set_label(df_4.iloc[:,1:].columns[j]+"-1")
    plt.ylabel("degree of gray")
    plt.legend()
    plt.show()

df_4 = pd.read_excel("高钾_4.xlsx")
for j in range(14):
    res = []
    x = np.linspace(0.1,2,100)
    bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
    data = df_4.iloc[:,1:].values
    for i in x:
        data = df_4.iloc[:,1:].values
        bias = np.random.random([data.shape[0],data.shape[1]])
        #bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
        #print(bias)
        data = data+bias
       # print(data)
        w=gray(data,0.5)
        SiO2 = w[j]
        res.append(SiO2)
       # print(np.sum(w))
    plt.ylim(0,1)
    draw, = plt.plot(res,'r')
    print(df_4.iloc[:1:].columns[0])
    draw.set_label(df_4.iloc[:,1:].columns[j]+"-1")
    plt.ylabel("degree of gray")
    plt.legend()
    plt.show()
    #res


# In[408]:


print(df_4.iloc[:,1:].columns[0])


# In[409]:


#灰色关联的超参数敏感性分析_二氧化硅——铅钡玻璃
df_4 = pd.read_excel("铅钡_4.xlsx")
res = []
x = np.linspace(0.1,2,100)
bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
data = df_4.iloc[:,1:].values
for i in x:
    data = df_4.iloc[:,1:].values
    bias = np.random.random([data.shape[0],data.shape[1]])
    #bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
    #print(bias)
    data = data+bias
   # print(data)
    w=gray(data,0.5)
    SiO2 = w[2]
    res.append(SiO2)
   # print(np.sum(w))
plt.ylim(0,1)
draw, = plt.plot(res,'r')
draw.set_label('SiO2-0')
plt.ylabel("degree of gray")
plt.legend()
plt.show()


# In[410]:


#灰色关联的超参数敏感性分析_氧化钾——铅钡玻璃
df_4 = pd.read_excel("铅钡_4.xlsx")
res = []
x = np.linspace(0.1,2,100)
bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
data = df_4.iloc[:,1:].values
for i in x:
    data = df_4.iloc[:,1:].values
    bias = np.random.random([data.shape[0],data.shape[1]])
    #bias = np.random.normal(loc=0.0, scale=1.0, size=[df_4.iloc[:,1:].values.shape[0],df_4.iloc[:,1:].values.shape[1]])
    #print(bias)
    data = data+bias
   # print(data)
    w=gray(data,0.5)
    SiO2 = w[2]
    res.append(SiO2)
   # print(np.sum(w))
plt.ylim(0,1)
draw, = plt.plot(res,'r')
draw.set_label('K2O-0')
plt.ylabel("degree of gray")
plt.legend()
plt.show()


# In[411]:


w = gray(df_4_qb.iloc[:,1:].values,0.5)#铅钡


# In[412]:


bias


# In[413]:


#导入数据——用于神经网络预测
# df_pred_mlp = pd.read_excel("神经网络.xlsx")
# # out = pd.get_dummies(df_pred_mlp.iloc[:,-2])
# # out.to_excel(excel_writer=r"颜色.xlsx")
# x_mlp = pd.get_dummies(df_pred_mlp.iloc[:,-4:])
# y_mlp = pd.get_dummies(df_pred_mlp.iloc[:,1:-4])

df_pred_mlp = pd.read_excel("train5.xlsx")
df_pred_mlp = df_train.dropna()
x_mlp = pd.get_dummies(df_train.iloc[:,-4:])
x_mlp = x_mlp.iloc[:,1:]
x_mlp = x_mlp.drop(labels=["颜色_绿"],axis=1)
y_mlp = df_pred_mlp.iloc[:,1:-5]

# df_pred_mlp = pd.read_excel("train5.xlsx")

# # out = pd.get_dummies(df_pred_mlp.iloc[:,-2])
# # out.to_excel(excel_writer=r"颜色.xlsx")
# x_mlp = pd.get_dummies(df_pred_mlp.iloc[:,-4:])
# y_mlp = pd.get_dummies(df_pred_mlp.iloc[:,1:-4])


# In[415]:


#基于神经网络对预测效果优化的完整过程
x_train_mlp,x_test_mlp,y_train_mlp,y_test_mlp = train_test_split(x_mlp,y_mlp,test_size=0.2)


model_mlp = MLPRegressor(
    hidden_layer_sizes=(18,2),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False, warm_start=False,  nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08,momentum=0.9)#momentum=0.9,

model_mlp.fit(x_train_mlp,y_train_mlp.iloc[:,0])
pred = model_mlp.predict(x_train_mlp)
mlp_score=model_mlp.score(x_train_mlp,y_train_mlp.iloc[:,0])

plt.plot(y_train_mlp.iloc[:,0].values, 'b')
plt.plot(pred, 'r')
plt.show()
print(mlp_score)


# In[ ]:




