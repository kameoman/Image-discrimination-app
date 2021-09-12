import streamlit as st
import sys
from PIL import Image
from keras.models import load_model
import numpy as np

uploaded_file = st.file_uploader("Choose an image...", type='jpg')
st.image(uploaded_file)

def main():
  image = Image.open(uploaded_file)
  image = image.resize((64, 64))
  # image.show()
  model = load_model("model.h5")
  np_image = np.array(image)
  np_image = np_image /255
  np_image = np_image[np.newaxis, :, :, :]
  result = model.predict(np_image)
  # print(result)
  if result[0][0] > result[0][1]:
    st.write("椎茸")
  else:
    st.write("ツキヨタケ")


if __name__ == "__main__":
    main()