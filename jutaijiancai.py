import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model

class jutaijiancai:
    def __init__(self,date,ahead):
        self.read_data()
        self.start = date
        self.ahead = ahead
        self.epoch = 50
        self.batch_size = 72
        self.hidden_cell = 50
        self.output_cell = 1

    def process(self):
        self.Preprocess_data()
        self.split_data()
        self.train_model()
        self.Output()

    def read_data(self):
        self.origin = pd.read_csv("jutaijiancai20180417-20180613.csv",encoding="ansi")
        self.originC5 = self.origin[["时间","TE506.PV","TE514.PV","TE708.PV","MMYW_AI1.PV"]]
        self.originC5.rename(columns={"时间":"time","TE506.PV":"te506","TE514.PV":"te514",
                                      "TE708.PV":"te708","MMYW_AI1.PV":"mmyw_ai1"},inplace=True)
        self.originC5.set_index("time",inplace=True)
        self.originC5.index = pd.to_datetime(self.originC5.index)
        self.decimals = pd.Series([2, 2, 2, 3], index=['te506', 'te514', 'te708','mmyw_ai1'])
        self.originC5 = self.originC5.round(self.decimals)
        self.originC5_interpolate = self.originC5.interpolate()

    def Preprocess_data(self):
        self.train_data = self.originC5_interpolate[self.start:]
        self.feature = self.train_data[["te514","te708","mmyw_ai1"]].shift(self.ahead).dropna()
        self.result = self.train_data[["te506"]]
        self.train_data = pd.concat([self.result,self.feature],axis=1).dropna()
        self.scale = StandardScaler()
        self.scale.fit(self.train_data)
        self.train_data_scale = self.scale.transform(self.train_data)

    def split_data(self):
        # split into train and test sets
        self.day_num = pd.DataFrame({"month": list(self.train_data.index.month), "day": list(self.train_data.index.day)}).drop_duplicates(["month", "day"]).shape[0]
        self.values = pd.DataFrame(self.train_data_scale,columns=["te506","te514","te708","mmyw_ai1"]).values
        self.n_train_sec = (self.day_num-1) * 3600 * 24
        self.train = self.values[:self.n_train_sec, :]
        self.test = self.values[self.n_train_sec:, :]
        # split into input and outputs
        self.train_X, self.train_y = self.train[:, 1:], self.train[:, :1]
        self.test_X, self.test_y = self.test[:, 1:], self.test[:, :1]
        # reshape input to be 3D [samples, timesteps, features]
        self.train_X = self.train_X.reshape((self.train_X.shape[0], 1, self.train_X.shape[1]))
        self.test_X = self.test_X.reshape((self.test_X.shape[0], 1, self.test_X.shape[1]))
        print(self.train_X.shape, self.train_y.shape, self.test_X.shape, self.test_y.shape)

    def train_model(self):
        # design network
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_cell, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dense(self.output_cell))
        self.model.compile(loss='mae', optimizer='Nadam')
        # fit network
        self.history = self.model.fit(self.train_X, self.train_y, epochs=self.epoch, batch_size=self.batch_size, validation_data=(self.test_X, self.test_y), verbose=2,
                            shuffle=False)
        plot_model(self.model, to_file='model hidden_cell %d output_cell %d.png'%(self.hidden_cell,self.output_cell))

        pass

    def Output(self):
        # plot history
        plt.figure(figsize=(30, 10))
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='test')
        plt.legend()
        plt.title("loss curve during LSTM learning process", size=20)
        plt.xlabel("epoch", size=20)
        plt.ylabel("mae", size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.savefig("%d  epochs Iteration.png" % (self.epoch))

        self.predict_y = self.model.predict(self.test_X).reshape(-1)
        self.predict = pd.DataFrame(self.scale.inverse_transform(pd.concat([pd.DataFrame(self.predict_y), pd.DataFrame(self.test_X.reshape(-1, 3))], axis=1))).iloc[:,0]
        self.real = pd.DataFrame(self.scale.inverse_transform(pd.concat([pd.DataFrame(self.test_y), pd.DataFrame(self.test_X.reshape(-1, 3))], axis=1))).iloc[:,0]
        self.mae = np.mean(np.abs(self.predict - self.real))
        self.mape = np.mean(np.abs(self.predict - self.real)/self.real)

        plt.figure(figsize=(30, 20))
        plt.plot(self.predict, label='predict')
        plt.plot(self.real, label='real')
        plt.title("LSTM  performance in test_set", size=20)
        plt.xlabel("time", size=20)
        plt.ylabel("℃", size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend(fontsize=20)
        plt.savefig("epoch %d_%d w.png"%(self.epoch,int(self.train_data.shape[0]/10000)))
        print("mae: %.2f,  mape: %.2f"%(self.mae,self.mape))


