import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/reptile.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file")

map_dict = {0: 'BlacklippedLizard',
            1:'GreenForestLizard',
            2:'CommonGardenLizard',
            3:'RoughhornedLizard',
            4:'SriLankanDayGecko',
            5:'SriLankanFlyingLizard',
            6:'SriLankanKangarooLizard',
            7:'BambooPitViper',
            8:'CheckeredKeelback',
            9:'CommonWolfSnake',
            10:'HumpnosedViper',
            11:'GreenPitViper',
            12:'CeylonKraitBungarusceylonicus',
            13:'RatSnake',
            14:'RussellsViper',
            15:'SriLankanFlyingSnake',
            16:'SriLankanKeelback'
            }


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        if prediction == 0 :
           st.text("Description The Black-lipped Lizard (Calotes nigrilabris) is a reptile species known for its striking appearance and unique features. With an average size of 30-40 centimeters, it showcases a vibrant combination of green and black coloration, complemented by a distinct black stripe on its upper lip. This lizard is classified as a species of least concern in terms of conservation status")
        else:
            st.title("else")            
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
