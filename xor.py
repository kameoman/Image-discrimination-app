from keras.models import Sequential
from keras.layers import Activation, Dense
import numpy as np

def main():
    model = Sequential()
    model.add(Dense(3, input_dim=2))
    model.add(Activation("sigmoid"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.summary()

    model.compile(
          # フィッティングのやり方を指定
          optimizer="adam",
          # 誤差を測定
          loss="mse",
          # フィッティングの途中で正しいか確認ができる
          metrics=["accuracy"]
    )
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0, 1, 1, 0])
    # epochsでフィッティングの回数を調整し、自分の欲しい値まで持っていく
    model.fit(x, y, epochs=5000)
    result = model.predict(x)
    print(result)


if __name__ == "__main__":
    main()