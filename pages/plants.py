import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("saved_model/plants.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file")

map_dict = { 0:'Black-Honey Shrub - Phyllanthus reticulatus',
             1:'Butterfly Pea - Clitoria ternatea',
             2:'Carissa Carandas (Karanda)',
             3:'Jasminum (Jasmine)',
             4:'kohomba - Azadirachta indica',
             5:'Mountain Knotgrass - Aerva lanata',
             6:'Muntingia Calabura (Jamaica Cherry-Gasagase)',
             7:'Prickly Chaff Flower - Achyranthes aspera',
             8:'Purple Fruited Pea Eggplant - Solanum trilobatum',
             9:'Purple Tephrosia - Tephrosia purpurea',
             10:'Balloon vine - Cardiospermum halicacabum',
             11:'Bellyache bush (Green) - Jatropha gossypiifolia',
             12:'blue porterweed - Stachytarpheta jamaicensis',
             13:'coatbuttons - Tridax procumbens',
             14:'Ivy Gourd - Coccinia grandis',
             15:'Santalum Album (Sandalwood)',
             16:'Shaggy button weed - Spermacoce hispida',
             17:'Velvet bean- Mucuna pruriens',
             18:'Wild Pepper - Solanum bahamense',
             19:'Ceylon satinwood (Chloroxylon swietenia)',
             20:'Hourglass tree (Humboldtia laurifolia)',
             21:'Serendib scaly tree fern (Cyathea contaminans)',
             22:'Sinharaja damba (Grewia damine)',
             23:'Boralu - Anisomeles malabarica',
             24:'Gammalu - Diospyros ovalifolia',
             25:'Divul - Diospyros malabarica',
             26:'Dun - Syzygium caryophyllatum',
             27:'Hora - Dipterocarpus zeylanicus',
             28:'Kahata - Ficus microcarpa',
             29:'Kekuna -Calophyllum inophyllum',
             30:'Kothala Himbutu - Salacia reticulata',
             31:'Meewana - Memecylon edule'
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
        # if(map_dict [prediction] == 'turmericfingers'){
        #     Grade = 1
        # }
        st.text(map_dict [prediction])
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
