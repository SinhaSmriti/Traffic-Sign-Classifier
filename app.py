import streamlit as st
import tensorflow as tf

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def test_on_img(img):
    data = []
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    x_test = np.array(data)
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return image, y_pred_classes

classes = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing vehicle over 3.5 tons',
    11:'Right-of-way intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'vehicle > 3.5 tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve left',
    20:'Dangerous curve right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End speed + passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End no passing vehicle > 3.5 tons'
}

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_data()
def load_model():
    model = tf.keras.models.load_model('./training/TSR.h5')
    return model
model = load_model()

st.title("Traffic Sign Classification")
st.write("Upload an image of traffic sign")

file = st.file_uploader("Choose an image: ", type=["jpg", "png", "jpeg"])

if file is not None:
    image, prediction = test_on_img(file)
    st.image(image, caption="Upload Image", use_column_width=True)
    s = [str(i) for i in prediction]
    a = int("".join(s))
    st.write(f"Predicted traffic sign is: {classes[a]}")
    fig, ax = plt.subplots()
    ax.imshow(image)
    st.pyplot(fig)
