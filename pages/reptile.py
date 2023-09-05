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
                    st.header('Black-lipped Lizard (Calotes nigrilabris)', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Class :blue[Reptilia]')
                    st.markdown(":green[Description:] The Black-lipped Lizard (Calotes nigrilabris) is a reptile species known for its striking appearance and unique features. With an average size of 30-40 centimeters, it showcases a vibrant combination of green and black coloration, complemented by a distinct black stripe on its upper lip. This lizard is classified as a species of least concern in terms of conservation status")
        if prediction == 10 :
                    st.header('Hump-nosed Viper', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[H. hypnale]')
                    st.text_area("Description : The Hump-nosed Viper (Hypnale spp.) is a venomous snake species found in the Sinharaja Forest. The Hump-nosed Viper is a medium-sized snake, averaging around 50-70 centimeters in length. It has a distinctive triangular-shaped head with a prominent hump on its nose. This species showcases various color variations, including shades of brown, gray, or olive, often accompanied by intricate patterns of zigzag lines or blotches along its body, aiding in its identification. It possesses keeled scales and vertical pupils")
        if prediction == 5  :
                    st.header('Sri Lankan Flying Lizard (Draco dussumieri)', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[D. dussumieri]')
                    st.text_area("Description : The Sri Lankan Flying Lizard (Draco dussumieri) is a remarkable reptile species found in the forests of Sri Lanka and parts of southern India. Known for its exceptional gliding ability, it possesses elongated ribs that support a wing-like patagium. With a length of around 25 centimeters, it showcases a striking combination of green and yellow hues, often adorned with intricate patterns. This lizard is considered of least concern in terms of conservation status, benefiting from its wide distribution and adaptability to various forest habitats")
        if prediction == 1 :
                    st.header('Green Forest Lizard (Calotes calotes)ard (Otocryptis wiegmanni)', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[C. calotes]')
                    st.text_area("Description : The Green Forest Lizard (Otocryptis wiegmanni) is a visually striking lizard species native to Sri Lanka. It features a brilliant green body with intricate patterns and scales. With a size of approximately 20-30 centimeters, this lizard is known for its agility and ability to climb trees. Currently, it is classified as a species of least concern in terms of conservation status.")
        if prediction == 2 :
                    st.header('Common Garden Lizard (Calotes versicolor)', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[C. versicolor]')
                    st.text_area("Description: The Common Garden Lizard (Calotes versicolor) is a well-known lizard species .It possesses a moderate size, averaging around 30-40 centimeters in length, and exhibits a vibrant range of colors. The males display an enchanting combination of green, yellow, and blue hues, while females and juveniles have a more subdued brown or olive coloration. This lizard is abundant in its range and is classified as a species of least concern in terms of conservation status.")
        if prediction == 3 :
                    st.header('Sri Lankan Day Gecko (Cnemaspis podihuna)', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[C. podihuna]')
                    st.text_area("Description: The Sri Lankan Day Gecko (Cnemaspis podihuna) is a captivating lizard species endemic to Sri Lanka. It is a small gecko with an average length of about 6-7 centimeters. Known for its vibrant colors, it features shades of green, yellow, and blue, with distinct patterns and markings on its body. This species is currently classified as vulnerable due to habitat loss and degradation caused by deforestation and urbanization. Conservation efforts are crucial to ensure the survival of this unique gecko.")
        if prediction == 4 :
                    st.header('Rough-horned Lizard (Ceratophora stoddartii)', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[C. stoddartii]')
                    st.text_area("Description: The Rough-horned Lizard (Ceratophora stoddartii) is a fascinating lizard species native to Sri Lanka. It derives its name from the unique horn-like projections on its head and along its body. With an average length of around 20 centimeters, it displays a brown or gray coloration with intricate patterns and textures. This lizard is considered endangered, facing threats such as habitat loss and illegal collection for the pet trade. Conservation efforts are crucial to protect and preserve this remarkable species.")
        if prediction == 6 :
                    st.header('Sri Lankan Kangaroo Lizard (Otocryptis wiegmanni)', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[O. weigmanni]')
                    st.text_area("Description: The Sri Lankan Kangaroo Lizard (Otocryptis wiegmanni) is a unique and endemic lizard species found in the forests of Sri Lanka. It gets its name from its remarkable hind legs that resemble those of a kangaroo. With an average length of 20-30 centimeters, it showcases a brown or olive coloration, often adorned with intricate patterns and scales. This lizard is currently classified as a species of least concern in terms of conservation status, benefiting from its localized habitat and relatively stable population")
        if prediction == 7 :
                    st.header('Bambo pit viper', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[C. gramineus]')
                    st.text_area("Venom:The Bamboo Pit Viper (Trimeresurus gramineus) is a venomous snake species.When threatened, it is aggressive and does not hesitate to bite. The venom is hemotoxic and neurotoxic.")
                    st.text_area("Description: The Bamboo Pit Viper is a medium-sized snake with an average length of around 70-90 centimeters. It displays striking coloration, ranging from vibrant shades of green to brown, allowing it to blend seamlessly into its arboreal habitat. This species is known for its triangular-shaped head, keeled scales, and a distinctive pattern of dark dorsal blotches or crossbands that run along its body, aiding in its identification.")
        if prediction == 8:
                    st.header('Checkered Keelback (Xenochrophis piscator)', divider='green')
                    st.subheader('Kingdom :blue[Animalia]')
                    st.subheader('Species :blue[F. piscator]')
                    st.text_area("Venom:Non venomus")
                    st.text_area("Description The Black-lipped Lizard (Calotes nigrilabris) is a reptile species known for its striking appearance and unique features. With an average size of 30-40 centimeters, it showcases a vibrant combination of green and black coloration, complemented by a distinct black stripe on its upper lip. This lizard is classified as a species of least concern in terms of conservation status")
        if prediction == 9 :
                    st.text_area("Description The Black-lipped Lizard (Calotes nigrilabris) is a reptile species known for its striking appearance and unique features. With an average size of 30-40 centimeters, it showcases a vibrant combination of green and black coloration, complemented by a distinct black stripe on its upper lip. This lizard is classified as a species of least concern in terms of conservation status")
        if prediction == 11 :
                    st.text_area("Description The Black-lipped Lizard (Calotes nigrilabris) is a reptile species known for its striking appearance and unique features. With an average size of 30-40 centimeters, it showcases a vibrant combination of green and black coloration, complemented by a distinct black stripe on its upper lip. This lizard is classified as a species of least concern in terms of conservation status")
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
