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
        if prediction == 4 :
                    st.header('Kohomba - Azadirachta indica', divider='green')
                    st.subheader('Height :blue[15–20 metres (49–66 ft), and rarely 35–40 metres (115–131 ft)]')
                    st.subheader('Leaves :blue[Imparipinnate compound, exstipulate, crowded; rachis 15-22.5 cm long]')
                    st.subheader('Related medicinal properties :blue[Reduces aggravation of kapha and pitta dosha, Natural antiseptic]')
                    st.subheader('Edible parts :blue[The tender shoots and flowers of the neem tree are eaten as a vegetable in India. A souplike dish called Veppampoo charu (Tamil) (translated as “neem flower rasam”) made of the flower of kohomba is prepared in Tamil Nadu. In Bengal, young kohomba leaves are fried in oil with tiny pieces of eggplant (brinjal). The dish is called neem begun bhaja and is the first item during a Bengali meal that acts as an appetizer. It is eaten with rice.]')
        if prediction == 3 :
                    st.header('Jasmine - Jasmine', divider='green')
                    st.subheader('Medicine :blue[Ancient Indian physicians such as Charaka and Sushruta used Jasminum grandiflorum for various medicinal purposes. This flower is also given a variety of names in India as it is used for different remedies. Parts of J. grandiflorum, including their sprouts and flowers (dried), have been used for prescriptions. This type of holistic medicine was used to treat various sicknesses such as dermatosis, coryza, and nasal haemorrhage]')
                    st.subheader('Description :blue[It is a scrambling deciduous shrub growing to 2–4 m tall. The leaves are opposite, 5–12 cm long, pinnate with 5–11 leaflets. The flowers are produced in open cymes, the individual flowers are white having corolla with a basal tube 13–25 mm long and five lobes 13–22 mm long.In Pakistan, it grows wild in the Salt Range and Rawalpindi District at 500–1500 m altitude.]')  
        if prediction == 4 :
                    st.header('Dun - Syzygium caryophyllatum', divider='green')
                    st.subheader('Useful Plant Parts :blue[Leaf,Bark,and Seed]')
                    st.subheader('Description :blue[Seeds are pergative,crushed leaves apply on burns and Boils]')  
        if prediction == 25 :
                    st.header('Divul - Diospyros malabarica')
                    st.subheader('Medical Usage :blue[The bark, leaves, flowers and fruits are much used in Ayurvedic medicine[ 317 ]. The fruit, when unripe, is said to be cold, light, and astringent; and to possesses anti-bacterial and anthelmintic activity[ 317 , 555 ]. It is used externally to heal sores and wounds[ 555 ]. When ripe, the fruit is beneficial in treating diarrhoea and dysentery; blood diseases; gonorrhoea and leprosy[ 317 ]. The fruit is also said to break fever, to be an antidote for snake poisoning, and to be demulcent[ 555 ]. The juice of the fresh bark is useful in the treatment of bilious fevers. Externall]')
                    st.subheader('Description :blue[Seeds are pergative,crushed leaves apply on burns and Boils]') 
        if prediction == 31 :
                    st.header('Meewana - Memecylon edule', divider='green')
                    st.subheader('Medical :blue[The bark is used to treat bruises[An infusion of the flowers is used to treat inflammation of the conjunctiva]')
                    st.text_area('Other uses:The leaves are rich in aluminium and have been used traditionally as a mordant for fixing the colour of dyes.A yellow and a crimson dye can be extracted from the leaves and flowers. It can be used for dyeing cottons and woven goods such as matsThe light brown wood is very hard, close-grained Used for house posts and building boatsAn excellent fuel, it also makes a good charcoalWe have no more specific information for this species, but in general the wood of Memecylon species is usually white to brown, very dense and heavy, sinking in water. It is also often very durable. Where the wood gets large enough it is often used traditionally for purposes such as poles, house posts, lumber and furnitureThe wood of many species in the genus has a high calorific value and is often favoured as a fuel and for making charcoal')
        st.text(map_dict [prediction])
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
