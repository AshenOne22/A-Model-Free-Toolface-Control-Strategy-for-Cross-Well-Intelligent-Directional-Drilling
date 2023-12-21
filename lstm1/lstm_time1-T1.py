import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# 将数据截取成3个一组的监督学习格式
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0:4]
        dataX.append(a)
        dataY.append(dataset[i:(i + look_back), 6])
    return dataX, dataY


def select_dataset(look_back=1):
    path = '../rl1/record/DDPG_Pendulum-v0'
    a_data_x, a_data_y = [], []
    for k in range(1):
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


def create_model(look_back, action_dim):
    # 构建 LSTM 网络
    model = keras.Sequential()
    model.add(layers.Dense(32))
    model.add(layers.Dense(action_dim))
    model.compile(loss='mse', optimizer='adam',
                  metrics=[keras.metrics.mean_squared_error, keras.metrics.mean_absolute_error])
    return model


def train_model(model_filepath, epochs_num=100, batch_size_num=64, load=False):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2)
    checkpoint = keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='auto')
    # callbacks_list = [early_stopping]
    callbacks_list=[checkpoint]
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
    x2 = x.reshape(-1, 4)

    x3 = scalerx.fit_transform(x2)
    y2 = scalery.fit_transform(y)

    x4 = x3.reshape(xshape)
    y3 = y2.reshape(yshape)

    return scalerx, scalery, x4, y3


def transform_test(scalerx, scalery, x, y):
    # 缩放数据

    # 数据缩到二维
    xshape = x.shape
    yshape = y.shape
    x2 = x.reshape(-1, 4)

    x3 = scalerx.transform(x2)
    y2 = scalery.transform(y)

    x4 = x3.reshape(xshape)
    y3 = y2.reshape(yshape)

    return x4, y3


def inverse_data(scalerx, scalery, x, y):
    # 缩放数据

    # 数据缩到二维
    xshape = x.shape
    yshape = y.shape
    x2 = x.reshape(-1, 4)

    x3 = scalerx.inverse_transform(x2)
    y2 = scalery.inverse_transform(y)

    x4 = x3.reshape(xshape)
    y3 = y2.reshape(yshape)

    return x4, y3


def draw_history(history):
    mean_absolute_error = history.history['mean_absolute_error']
    val_mean_absolute_error = history.history['val_mean_absolute_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, mean_absolute_error, 'b', label='训练集')
    plt.plot(epochs, val_mean_absolute_error, 'g', label='验证集')
    # plt.title('Training and validation mean_absolute_error')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.savefig('image/mean_absolute_error.jpg')
    plt.show()
    plt.plot(epochs, loss, 'b', label='训练集')
    plt.plot(epochs, val_loss, 'g', label='验证集')
    # plt.title('Training and validation loss')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.savefig('image/loss.jpg')
    plt.show()


def draw_predict(scalerx, scalery, x, y):
    y_pre = model.predict(x)
    # 画图
    plt.figure()
    plt.plot(y[:100], label='y')
    plt.plot(y_pre[:100], label='yPredict')
    plt.legend()
    plt.title('testy')
    plt.savefig(os.path.join(imagepath, 'testy.jpg'))
    plt.show()

    # 数据还原再画图
    ori_y = scalery.inverse_transform(y)
    ori_y_pre = scalery.inverse_transform(y_pre)
    # 画图
    plt.figure()
    plt.plot(ori_y[:100], label='y')
    plt.plot(ori_y_pre[:100], label='yPredict')
    plt.legend()
    plt.title('ori_testy')
    plt.savefig(os.path.join(imagepath, 'ori_testy.jpg'))
    plt.show()

def draw_0(y_pre, y):
    # y_pre = model.predict(x)
    # 画图
    plt.figure()
    plt.plot(y[:100], label='y')
    plt.plot(y_pre[:100], label='yPredict')
    plt.legend()
    plt.title('testy')
    plt.savefig(os.path.join(imagepath, 'testy.jpg'))
    plt.show()


if __name__ == '__main__':
    look_back = 1
    state_dim = 4
    action_dim = 1
    action_range = [-2, 2]
    model_path = 'model-T1'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    imagepath = 'image-T1'
    if not os.path.exists(imagepath):
        os.makedirs(imagepath)

    data_x, data_y = select_dataset(look_back)

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=1)

    # # 缩放数据
    # scalerx, scalery, sx_train, sy_train = transform_train(x_train, y_train)
    # sx_test, sy_test = transform_test(scalerx, scalery, x_test, y_test)

    # 构建模型
    model = create_model(look_back, action_dim)
    # 训练模型
    history = train_model(model_path, epochs_num=100, load=False)

    # 对训练数据的Y进行预测
    trainPredict = model.predict(x_train)
    # 对测试数据的Y进行预测
    testPredict = model.predict(x_test)
    # # 对数据进行逆缩放
    # ori_testPredict = scalery.inverse_transform(testPredict)

    # # 画图
    # draw_predict(scalerx, scalery, x_test, y_test)
    draw_history(history)
    draw_0(testPredict,y_test)
