import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
from utils import load_and_prep, get_classes

#import SessionState
@st.cache(suppress_st_warning=True)
def body():
    pass

def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df
class_names = get_classes()
def main():
    

    st.title("Food😋 Vision Big 🤔🍨🍕 ")
    st.header("Project idea💡")
    st.write("CNN Image Classification Models which identifies the food in your image.To know more about this app and get all resources [click here](https://github.com) ")

    choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    (
        "Home",
     # original 10 classes
     "Model 2 (101 food classes)",) # original 10 classes + donuts

    )

# Model choice logic
    if choose_model == "Home":
        st.title("pick Model from slide bar")

                    
    else:
        st.write("Model Name- 101 classes")
        st.header("Upload Food Image File")
        file = st.file_uploader(label="",type=["jpg","jpeg","png"])
        model = tf.keras.models.load_model("./Model/my_model.h5")
        if not file:
            st.warning("Please upload an image")
            st.stop()

        else:
            image = file.read()
            st.image(image, use_column_width=True)
            pred_button = st.button("Predict")

        if pred_button:
            pred_class, pred_conf, df = predicting(image, model)
            st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf*100:.2f}%')



    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.header("")
    st.sidebar.markdown("_Created by_  **Manish Agarwal**")
    st.sidebar.markdown(body="""
    <th style="border:None"><a href="https://twitter.com/Manisha27493225" target="blank"><img align="center" src="https://bit.ly/3wK17I6" alt="gaurxvreddy" height="20" width="20" /></a></th>
    <th style="border:None"><a href="https://www.linkedin.com/in/manish-agarwal-2782511ba/" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="gauravreddy08" height="20" width="20" /></a></th>
    <th style="border:None"><a href="https://github.com/Manish06097" target="blank"><img align="center" src="https://cdn-icons.flaticon.com/png/512/4926/premium/4926624.png?token=exp=1655556983~hmac=abc4ff34b3149e7cfc43c1365808ad5d" alt="16034820" height="20" width="20" /></a></th>
    <th style="border:None"><a href="https://www.instagram.com/re_creator_sky/" target="blank"><img align="center" src="https://bit.ly/3oZABHZ" alt="gaurxv_reddy" height="20" width="20" /></a></th>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()


