"""
原始数据训练
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# 将数据截取成3个一组的监督学习格式
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0:in_dim]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0:out_dim])
    return dataX, dataY


def select_dataset(look_back=1):
    """
    构造数据集
    :param look_back:
    :return:
    """
    path = '../rl1/record/DDPG_Pendulum-v0'
    a_data_x, a_data_y = [], []
    for k in range(100):
        file_name = str(k) + '.csv'
        filepath = os.path.join(path, file_name)
        col_l = 'state0,state1,state2,action,state_0,state_1,state_2'
        col_l = col_l.split(',')
        df = pd.read_csv(filepath, usecols=col_l)
        data = df.values
        dataset = data.astype('float32')

        data_x, data_y = create_dataset(dataset, look_back)
        a_data_x += data_x
        a_data_y += data_y

    a2_data_x = np.array(a_data_x)
    a2_data_y = np.array(a_data_y)

    return a2_data_x, a2_data_y


def create_model(look_back, out_dim):
    # 构建 LSTM 网络
    model = keras.Sequential()
    model.add(layers.LSTM(128, input_shape=(look_back, in_dim), return_sequences=True))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(out_dim))
    model.compile(loss='mse', optimizer='adam',
                  metrics=[keras.metrics.mean_squared_error, keras.metrics.mean_absolute_error])
    return model


def train_model(model_filepath, epochs_num=100, batch_size_num=64, load=False):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2)
    callbacks_list = [early_stopping]
    filename = 'lstm1.h5'
    model_filepath = os.path.join(model_filepath, filename)
    if os.path.exists(model_filepath) and load:
        model.load_weights(model_filepath)
        print("checkpoint_loaded")
    history = model.fit(x_train, y_train, epochs=epochs_num, batch_size=batch_size_num,
                        validation_data=(x_test, y_test), callbacks=callbacks_list, verbose=2, shuffle=True)
    scores = model.evaluate(x_test, y_test, verbose=1)
    # 保存模型
    model.save(model_filepath)

    print(model.metrics_names, ':', scores)
    # print("数据训练结束，权重最优模型已保存。蓝色曲线代表预测值，红色曲线代表真实值。")

    return history


def transform_train(x, y):
    # 缩放数据
    scalerx = MinMaxScaler(feature_range=(0, 1))
    scalery = MinMaxScaler(feature_range=(0, 1))

    # 数据缩到二维
    xshape = x.shape
    yshape = y.shape
    x2 = x.reshape(-1, in_dim)
    y2 = y.reshape(-1, out_dim)

    x3 = scalerx.fit_transform(x2)
    y3 = scalery.fit_transform(y2)

    x4 = x3.reshape(xshape)
    y4 = y3.reshape(yshape)

    return scalerx, scalery, x4, y4


def transform_test(scalerx, scalery, x, y):
    # 缩放数据

    # 数据缩到二维
    xshape = x.shape
    yshape = y.shape
    x2 = x.reshape(-1, in_dim)
    y2 = y.reshape(-1, out_dim)

    x3 = scalerx.transform(x2)
    y3 = scalery.transform(y2)

    x4 = x3.reshape(xshape)
    y4 = y3.reshape(yshape)

    return x4, y4


def inverse_data(scalerx, scalery, x, y):
    # 缩放数据

    # 数据缩到二维
    xshape = x.shape
    yshape = y.shape
    x2 = x.reshape(-1, in_dim)
    y2 = y.reshape(-1, out_dim)

    x3 = scalerx.inverse_transform(x2)
    y3 = scalery.inverse_transform(y2)

    x4 = x3.reshape(xshape)
    y4 = y3.reshape(yshape)

    return x4, y4


def draw_history(history):
    mean_absolute_error = history.history['mean_absolute_error']
    val_mean_absolute_error = history.history['val_mean_absolute_error']
    mean_squared_error = history.history['mean_squared_error']
    val_mean_squared_error = history.history['val_mean_squared_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, mean_absolute_error, 'b', label='Training mean_absolute_error')
    plt.plot(epochs, val_mean_absolute_error, 'g', label='Validation mean_absolute_error')
    plt.title('Training and validation mean_absolute_error')
    plt.xlabel('Epochs')
    plt.ylabel('mean_absolute_error')
    plt.legend()
    plt.savefig(os.path.join(imagePath, 'mean_absolute_error.jpg'))
    plt.show()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(imagePath, 'loss.jpg'))
    plt.show()
    plt.plot(epochs, mean_squared_error, 'b', label='Training mean_squared_error')
    plt.plot(epochs, val_mean_squared_error, 'g', label='Validation mean_squared_error')
    plt.title('Training and validation mean_squared_error')
    plt.xlabel('Epochs')
    plt.ylabel('mean_squared_error')
    plt.legend()
    plt.savefig(os.path.join(imagePath, 'mean_squared_error.jpg'))
    plt.show()


def draw_predict(scalerx, scalery, x, y):
    y_pre = model.predict(x)
    dim = y.shape[-1]
    # 画图
    plt.figure()
    for i in range(dim):
        label_y = 'y' + str(i)
        label_ypre = 'yPredict' + str(i)
        plt.plot(y[:100, i], label=label_y)
        plt.plot(y_pre[:100, i], label=label_ypre)
    plt.legend()
    plt.title('testy')
    plt.savefig(os.path.join(imagePath, 'testy.jpg'))
    plt.show()

    # 数据还原再画图
    ori_x, ori_y = inverse_data(scalerx, scalery, x, y)
    ori_x, ori_y_pre = inverse_data(scalerx, scalery, x, y_pre)
    # 画图
    plt.figure()
    for i in range(dim):
        label_ori_y = 'ori_y' + str(i)
        label_ori_ypre = 'ori_yPredict' + str(i)
        plt.plot(ori_y[:100, i], label=label_ori_y)
        plt.plot(ori_y_pre[:100, i], label=label_ori_ypre)
    plt.legend()
    plt.title('ori_testy')
    plt.savefig(os.path.join(imagePath, 'ori_testy.jpg'))
    plt.show()


def draw_predict_ori(x, y):
    """
    训练数据未标准化
    :param x:
    :param y:
    :return:
    """
    y_pre = model.predict(x)
    dim = y.shape[-1]
    # 画图
    plt.figure()
    for i in range(dim):
        label_y = 'y' + str(i)
        label_ypre = 'yPredict' + str(i)
        plt.plot(y[:100, i], label=label_y)
        plt.plot(y_pre[:100, i], label=label_ypre)
    plt.legend()
    plt.title('testy')
    plt.savefig(os.path.join(imagePath, 'testy.jpg'))
    plt.show()


if __name__ == '__main__':
    dataform = 'ori'
    look_back = 1
    in_dim = 4
    out_dim = 3
    action_range = [-2, 2]
    path = '_'.join(['timestep', str(look_back), 'outdim', str(out_dim), dataform])
    model_path = os.path.join('model', path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    imagePath = os.path.join('image', path)
    if not os.path.exists(imagePath):
        os.makedirs(imagePath)
    # 数据集
    data_x, data_y = select_dataset(look_back)
    # 划分训练集
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1)

    # # 保存原始值
    # ori_x_train, ori_x_test, ori_y_train, ori_y_test = x_train, x_test, y_train, y_test
    # # 缩放数据
    # scalerx, scalery, x_train, y_train = transform_train(x_train, y_train)
    # x_test, y_test = transform_test(scalerx, scalery, x_test, y_test)

    # 构建模型
    model = create_model(look_back, out_dim)
    # # 对测试数据的Y进行预测
    # testPredict = model.predict(x_test)
    # 训练模型
    history = train_model(model_path, epochs_num=200, load=False)

    # 对训练数据的Y进行预测
    trainPredict = model.predict(x_train)
    # 对测试数据的Y进行预测
    testPredict = model.predict(x_test)
    # 对数据进行逆缩放

    # 画图
    draw_predict_ori(x_test,y_test)
    # draw_predict(scalerx, scalery, x_test, y_test)
    draw_history(history)
