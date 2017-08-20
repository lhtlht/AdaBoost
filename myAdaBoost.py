#-*- coding:utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(111)

'''
基础模型
data    :输入训练数据
Dt      :第t个基础模型对应的训练样本的权重
model   :找到一个阈值，以该阈值为分类边界进行分类
'''
def base_model(data,Dt):
    m = data.shape[0]
    pred = []
    pos  = None
    mark = None
    min_err = np.inf
    for j in range(m):
        pred_temp = []
        sub_mark = None
        lsum = np.sum(data[:j,1]) #计算左值
        rsum = np.sum(data[j:,1]) #计算右值
        if lsum < rsum:
            sub_mark = -1
            pred_temp.extend([-1.]*(j))
            pred_temp.extend([1.]*(m-j))
        else:
            sub_mark = 1
            pred_temp.extend([1.]*(j))
            pred_temp.extend([-1.]*(m-j))
        err = np.sum( 1 * (data[:,1] != pred_temp)*Dt ) #计算误差
        if err < min_err:
            min_err = err
            pos = (data[:,0][j-1] + data[:,0][j])/2
            mark = sub_mark
            pred = pred_temp[:]
    model = [pos,mark,min_err]
    return model,pred




def adaboost(data):
    models = []
    m = data.shape[0]       #训练样本大小
    D = np.zeros(m) + 1.0/m #初始化训练样本权值分布
    T = 3                   #训练轮数
    y = data[:,-1]

    for t in range(T):
        Dt = D[:]
        model,y_ = base_model(data,Dt)
        print model
        errt = model[-1]
        #更新alpha值
        alpha = 0.5*np.log((1-errt)/errt)
        Zt = np.sum([Dt[i]*np.exp(-alpha*y[i]*y_[i]) for i in range(m)])
        #更新权值
        D  = np.array([Dt[i]*np.exp(-alpha*y[i]*y_[i]) for i in range(m)]) / Zt
        models.append([model,alpha])
    return models
'''
组合T个基础模型，构建最终的模型并进行预测
'''
def adaboost_prediction(models,X):
    pred_ = []
    for x in X:
        result = 0
        for base in models:
            alpha_ = base[1]
            if x[0] > base[0][0]:
                result -= base[0][1] * alpha_
            else:
                result += base[0][1] * alpha_
        pred_.append(np.sign(result))
    return pred_


if __name__ == "__main__":
    data = np.array([[0,1],[1,1],[2,1],[3,-1],[4,-1],[5,-1],[6,1],[7,1],[8,1],[9,-1]],
                    dtype=np.float32)
    plt.scatter(data[:,0],data[:,1],c=data[:,1],cmap=plt.cm.Paired)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()

    models = adaboost(data)
    X = data
    Y = data[:,-1]
    Y_=adaboost_prediction(models,X)
    acc = np.sum(1*(Y==Y_)) / float(len(X))
    print acc