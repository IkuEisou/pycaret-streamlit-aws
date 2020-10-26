from logging import debug
from pycaret.regression import *
from pycaret.datasets import get_data
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import requests
import os
import fnmatch
from pathlib import Path
import math

IMG_PATH = 'public/static/img/'
MODEL_PATH = 'volume/models/'
Features = {
    'insurance': ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
    'diamond': ['Carat', 'Weight', 'Cut', 'Color',
                'Clarity', 'Polish', 'Symmetry', 'Report'],
    'iris': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
}
LabelEncoded = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.splitext(name)[0])
    return result


def trainModel(dType, dataset, targetVal, target):
    if st.button("作成"):
        url = 'http://127.0.0.1:5000/automl'
        myobj = {'type': dType, 'target': targetVal, 'dataset': dataset}
        chunk_size = 1024
        'Waiting for training model...'
        r = requests.post(url, data=myobj, stream=True)
        content_size = int(r.headers['content-length'])
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        fileName = dataset+'_'+dt_string+'.pkl'
        with open(MODEL_PATH+dataset+'/'+fileName, "wb") as pkl:
            # Add a placeholder
            latest_iteration = st.empty()
            'Downloading the model...'
            bar = st.progress(0)
            size = 0
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    pkl.write(chunk)
                    size = len(chunk)+size
                    # Update the progress bar with each iteration.
                    latest_iteration.text(
                        f'Iteration {int(size/content_size)*100}')
                    bar.progress(int(size/content_size)*100)
                    time.sleep(0.1)

        st.success(
            'The model of predicting {} from {} has been finished'.format(target, dataset))


def run():
    from PIL import Image
    image = Image.open(IMG_PATH+'logoMilizePycaret.png')
    image_milize = Image.open(IMG_PATH+'milize_logo.png')
    irisImage = Image.open(IMG_PATH+'iris-machinelearning.jpg')
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .sidebar .sidebar-content {
                background-image: linear-gradient(#008a07,#008a07);
                color: white;
            }
            .main{
                background-color: #111111;
                label{
                    color: white;
                }
            }
            .main label{
                color: white;
            }
            .stNumberInput div.input-container div.controls .control{
                background-color:#008a07;
            }
            h1{
                color: white
            }
            .stSelectbox label{
                color: white
            }
            .st-bp{
                color: white;
            }
            .st-d9{
                background-color:#1e3216;
                border: 0;
            }
            .st-bu{
                background-color:#1e3216;
                border: 0;
            }
            .st-al{
                color:white;
            }
            .streamlit-button.primary-button{
                background-color:#008a07;
                color:white;
                border: 0;
            }
            .stButton{
                text-align: center;
            }
            .stAlert{
                bottom: 10px;
            }
            .st-ej {
                color: white;
            }
            </style>
            """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.image(image, use_column_width=False, width=800)
    st.sidebar.image(image_milize, width=200)
    add_selectbox = st.sidebar.selectbox(
        "予測方法を選択",
        ("Online", "Batch"))
    # st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.info('Copyright © Milize 2020')

    st.title("AutoML予測アプリ")
    dType = st.selectbox('Type', ['Regression', 'Classification'])
    ds = ''
    if(dType == 'Regression'):
        ds = ['insurance']
    elif(dType == 'Classification'):
        ds = ['iris']
    dataset = st.selectbox('Dataset', ds)
    targetVal = ''
    if dataset == 'insurance':
        targetVal = 'charges'
    elif dataset == 'iris':
        targetVal = 'species'
    Path(MODEL_PATH+dataset).mkdir(parents=True, exist_ok=True)
    models = find('*.pkl', MODEL_PATH+dataset+'/')
    target = st.selectbox('Target', [targetVal])
    if len(models) == 0:
        st.warning("モデルはありません。新規作成してください。")
        trainModel(dType, dataset, targetVal, target)
    elif add_selectbox == 'Online':
        modelName = st.selectbox('Model', models)
        input_dict = {}
        if dataset == 'insurance':
            age = st.number_input('Age', min_value=1, max_value=100, value=25)
            sex = st.selectbox('Sex', ['male', 'female'])
            bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
            children = st.selectbox(
                'Children', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            if st.checkbox('Smoker'):
                smoker = 'yes'
            else:
                smoker = 'no'
            region = st.selectbox(
                'Region', ['southwest', 'northwest', 'northeast', 'southeast'])
            input_dict = {'age': age, 'sex': sex, 'bmi': bmi,
                          'children': children, 'smoker': smoker, 'region': region}
        elif dataset == 'iris':
            sepal_length = st.number_input(
                'sepal_length', min_value=1.0, max_value=100.0, value=5.1)
            sepal_width = st.number_input(
                'sepal_width', min_value=1.0, max_value=50.0, value=3.5)
            petal_length = st.number_input(
                'petal_length', min_value=1.0, max_value=100.0, value=1.4)
            petal_width = st.number_input(
                'petal_width', min_value=0.0, max_value=50.0, value=0.2)
            input_dict = {'sepal_length': sepal_length, 'sepal_width': sepal_width,
                          'petal_length': petal_length, 'petal_width': petal_width
                          }
        output = ""
        input_df = pd.DataFrame([input_dict])

        model = load_model(MODEL_PATH+dataset+'/'+modelName)
        if st.button("予測する"):
            output = predict(model=model, input_df=input_df)
            if dataset == 'insurance':
                output = '保険料予測は {}'.format(
                    str(math.ceil(output*104.86)) + '円')
            elif dataset == 'iris':
                output = 'This is '+LabelEncoded[output]
            st.success(output)
            if dataset == 'iris':
                st.image(irisImage, width=500)

    elif add_selectbox == 'Batch':
        modelName = sex = st.selectbox('Model', models)
        file_upload = st.file_uploader(
            "Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            print(file_upload)
            data = pd.read_csv(file_upload)
            model = load_model(MODEL_PATH+dataset+'/'+modelName)
            predictions = predict_model(estimator=model, data=data).rename(
                columns={'Label': target})
            predictions[target].replace(
                {0: LabelEncoded[0], 1: LabelEncoded[1], 2: LabelEncoded[2]}, inplace=True)

            st.write(predictions)
            if dataset == 'iris':
                st.image(irisImage, width=500)


if __name__ == '__main__':
    run()
