from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten

def main():
    model = Sequential()
    # 畳み込み層
    model.add(Conv2D(64,(3,3), input_shape=(64,64,3)))
    model.add(Activation("relu"))
    # MaxPooling画像の特定の領域から最大値を抽出する操作
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(2))
    model.add(Activation("softmax"))
    model.summary()


if __name__ == "__main__":
    main()