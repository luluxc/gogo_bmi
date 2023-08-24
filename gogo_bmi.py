# filename = '/home/appuser/venv/lib/python3.9/site-packages/keras_vggface/models.py'
filename = 'usr/local/lib/python3.9/dist-packages/keras_vggface/models.py'
text = open(filename).read()
open(filename, 'w+').write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
import keras.utils as image
from keras_vggface import VGGFace
from keras.models import Model
from keras_vggface.utils import preprocess_input
import numpy as np
import time
import io
import joblib
import pandas as pd
import av
import logging
import os
from turn import get_ice_servers
import threading
from typing import Union
import whatimage
import pyheif


#def pearson_corr(y_test, y_pred):
#  corr = tfp.stats.correlation(y_test, y_pred)
#  return corr

# def custom_object_scope(custom_objects):
#   return tf.keras.utils.CustomObjectScope(custom_objects)

# with custom_object_scope({'pearson_corr': pearson_corr}):
#   model = load_model('My_model_vgg16.h5')

#model = load_model('/content/gdrive/MyDrive/Colab Notebooks/My_BMI/My_model_vgg16.h5', compile=False)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@st.cache_data
def calculator(height, weight):
  return 730 * weight / height**2
  
def change_photo_state():
  st.session_state['photo'] = 'Done'

@st.cache_resource(show_spinner=False)
def load_svr():
    return joblib.load('senet_svr_model.pkl')

svr_model = load_svr()
  
@st.cache_resource(show_spinner=False)
def load_vggface():
    vggface = VGGFace(model='senet50', input_shape=(224, 224, 3), pooling='avg')
    return Model(inputs=vggface.input, outputs=vggface.get_layer('avg_pool').output)

vggface_model = load_vggface()

@st.cache_resource(show_spinner=False)
def get_resnet_feature(img):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, version=2) 
    fc6_feature = vggface_model.predict(img)
    return fc6_feature

def predict_bmi(frame):
    pred_bmi = []
    faces = faceCascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor = 1.15,minNeighbors = 5,minSize = (30,30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        image = frame[y:y+h, x:x+w]
        img = image.copy()
        img = cv2.resize(img, (224, 224))
        img = np.array(img).astype(np.float64)
        features = get_resnet_feature(img)
        flat_feature = np.reshape(features, (features.shape[0], features.shape[-1]))
        preds = svr_model.predict(flat_feature)
        pred_bmi.append(preds[0])
        cv2.putText(frame, f'BMI: {preds}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return pred_bmi, frame
 
def prepare_download(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    image_bytes = buf.getvalue()
    return image_bytes

# def allowed_file(filename):
#     ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# @st.cache_data
# def decodeImage(bytesIo):
#     fmt = whatimage.identify_image(bytesIo)
#     if fmt in ['heic', 'avif']:
#       i = pyheif.read_heif(bytesIo)
#       # Convert to other file format like jpeg
#       pi = Image.frombytes(mode=i.mode, size=i.size, data=i.data)
#       pi.save('new.jpg', format="jpeg")
#       return 'new.jpg'
 
class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.out_image = None
        self.pred_bmi = []

    def recv(self, frame):
        frm = frame.to_ndarray(format='bgr24')
        pred_bmi, frame_with_bmi = predict_bmi(frm)
        with self.frame_lock:
            self.out_image = frame_with_bmi
            self.pred_bmi = pred_bmi

        return av.VideoFrame.from_ndarray(frame_with_bmi, format='bgr24') 

def main():
  if 'photo' not in st.session_state:
    st.session_state['photo'] = 'Not done'
  st.set_page_config(layout="centered", page_icon='random')
  st.markdown("""
  <style>
  .big-font {
      font-size:80px !important;
  }
  </style>
  """, unsafe_allow_html=True)
  st.balloons()
  st.markdown('<p class="big-font">BMI Prediction 📸</p>', unsafe_allow_html=True)
  bmi_img = Image.open('bmi.jpeg')
  st.image(bmi_img)
  #st.title('*BMI prediction 📸*')
  st.write('Body Mass Index(BMI) estimates the total body fat and assesses the risks for diseases related to increase body fat. A higher BMI may indicate higher risk of developing many diseases.')
  st.write('*Since we only have the access to your face feature, the estimated value is biased')
  
  st.subheader('Try this live BMI predictor 😄')
  webrtc_streamer(key="example",video_transformer_factory=VideoProcessor,rtc_configuration={'iceServers': get_ice_servers()},sendback_audio=False)
  
  col2, col3 = st.columns([2,1])
  upload_img = col3.file_uploader('Upload a photo 🖼', type=['png', 'jpeg', 'jpg'], on_change=change_photo_state)
  file_image = col2.camera_input('Take a pic of you 😊', on_change=change_photo_state)         

  if st.session_state['photo'] == 'Done':
    process_bar3 = col3.progress(0, text='🏃‍♀️')
    process_bar2 = col2.progress(0, text='🏃')

    if file_image:
      for process in range(100):
        time.sleep(0.01)
        process_bar2.progress(process+1)
      col2.success('Taken the photo sucessfully!')
      file_image = np.array(Image.open(file_image))
      pred_camera = predict_bmi(file_image)[0]
      if len(pred_camera) == 0:
        col2.warning('No face detected, please take a photo again.')
      ready_cam = Image.fromarray(file_image)
      col2.image(ready_cam, clamp=True)
      # Convert the PIL Image to bytes
      download_cam = prepare_download(ready_cam)
      col3.divider()
      col3.write('Download the predicted image if you want!👇')
      download_img = col3.download_button(
        label=':black[Download image]', 
        data=download_cam,
        file_name='BMI_image_camera.png',
        mime="image/png",
        use_container_width=True)
    elif upload_img:
      for process in range(100):
        time.sleep(0.01)
        process_bar3.progress(process+1)
      col3.success('Uploaded the photo sucessfully!')
#       new_img = decodeImage(upload_img.getvalue())
#       if new_img:
#          upload_img = Image.open(new_img)
      upload_img = Image.open(upload_img)
      upload_img = np.array(upload_img.convert('RGB'))
      pred_upload = predict_bmi(upload_img)[0]
      if len(pred_upload) == 0:
        col2.warning('No face detected, please upload a photo again.')
      ready_upload = Image.fromarray(upload_img)
      col2.image(ready_upload, clamp=True)
      # Convert the PIL Image to bytes
      download_img = prepare_download(ready_upload)
      col3.write('Download the predicted image if you want!')
      download_img = col3.download_button(
        label='Download image', 
        data=download_img,
        file_name='BMI_image_uploaded.png',
        mime="image/png")

  
  expander_col2 = col2.expander('What are the health consequences of having a high or low BMI?')
  expander_col2.write('Obesity carries significant health hazards, whereas maintaining a healthy weight is a preventative measure against illnesses and cardiovascular difficulties. People with a BMI of more than 30 are more likely to have problems such as:')
  expander_col2.write('                       *hypertension')
  expander_col2.write('                       *diabetes type 2')
  expander_col2.write('                       *coronary artery disease (CAD)')
  expander_col2.write('                       *Arthritis, certain forms of cancer, and respiratory issues')
  expander_col2.write('Even a healthy BMI isn’t a guarantee of good health. Malnutrition, osteoporosis, anemia, and a range of other issues can develop from nutrient insufficiency in those with a BMI below 18.5. Low BMI could indicate hormonal, intestinal, or other issues.')
  
  index = {'BMI':['16 ~ 18.5', '18.5 ~ 25', '25 ~ 30', '30 ~ 35', '35 ~ 40', '40~'],
           'WEIGHT STATUS':['Underweight', 'Normal', 'Overweight', 'Moderately obese', 'Severely obese', 'Very severely obese']}
  df = pd.DataFrame(data=index)
  hide_table_row_index = """<style>
                            thead tr th:first-child {display:none}
                            tbody th {display:none}
                            </style>"""
  col3.markdown(hide_table_row_index, unsafe_allow_html=True)
  col3.table(df)
  expander = col3.expander('BMI Index')
  expander.write('The table above shows the standard weight status categories based on BMI for people ages 20 and older.')
  expander.write('(Note: This is just the reference, please consult professionals for more health advices.)')
  
  col3.title('BMI calculator')
  cal = col3.container()
  with cal:
    feet = col3.number_input(label='Height(feet)')
    inch = col3.number_input(label='Height(inches)')
    weight = col3.number_input(label='Weight(pounds)')
    if col3.button('Calculate BMI'):
      if feet == 0.0:
        col3.warning('Please fill in your height(feet)')
      elif weight == 0.0:
        col3.warning('Please fill in your weight(pounds)')
      else:
        height = feet * 12 + inch
        score = calculator(height, weight)
        col3.success(f'Your BMI value is: {score}')
      
if __name__=='__main__':
    main()
