from keras.models import Sequential
from keras.layers import Activation, Dense

def main():
    model = Sequential()
    model.add(Dense(3, input_dim=2))
    model.add(Activation("sigmoid"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.summary()


if __name__ == "__main__":
    main()