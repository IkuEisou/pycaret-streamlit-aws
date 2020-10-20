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

IMG_PATH = 'public/static/img/'


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


def trainModel():
    dType = st.selectbox('Type', ['Classification', 'Regression'])
    ds = ''
    if(dType == 'Regression'):
        ds = ['insurance', 'house']
    elif(dType == 'Classification'):
        ds = ['credit', 'bank']
    dataset = st.selectbox('Dataset', ds)
    targetVal = ''
    if dataset == 'insurance':
        targetVal = 'charges'
    elif dataset == 'house':
        targetVal = 'SalePrice'
    elif dataset == 'credit':
        targetVal = 'default'
    elif dataset == 'bank':
        targetVal = 'deposit'

    target = st.selectbox('Target', [targetVal])
    if st.button("作成"):
        url = 'http://127.0.0.1:5000/automl'
        myobj = {'target': targetVal, 'dataset': dataset}
        chunk_size = 1024
        'Waiting for training model...'
        r = requests.post(url, data=myobj, stream=True)
        content_size = int(r.headers['content-length'])
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")
        fileName = 'best_model_'+dataset+'_'+dt_string+'.pkl'
        with open(fileName, "wb") as pkl:
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
    image = Image.open(IMG_PATH+'logo.png')
    image_hospital = Image.open(IMG_PATH+'hospital.jpg')
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.image(image, use_column_width=False)
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))
    # st.sidebar.info('This app is created to predict patient hospital charges')
    # st.sidebar.success('https://milize.co.jp')
    st.sidebar.image(image_hospital)
    st.title("Milize AutoML Demo App")

    models = find('*.pkl', './')
    if len(models) == 0:
        st.warning("モデルはありません。新規作成してください。")
        trainModel()
    elif add_selectbox == 'Online':
        modelName = sex = st.selectbox('Model', models)
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox(
            'Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output = ""

        input_dict = {'age': age, 'sex': sex, 'bmi': bmi,
                      'children': children, 'smoker': smoker, 'region': region}
        input_df = pd.DataFrame([input_dict])
        model = load_model(modelName)
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = '$' + str(output)

        st.success('The output is {}'.format(output))

    elif add_selectbox == 'Batch':
        modelName = sex = st.selectbox('Model', models)
        file_upload = st.file_uploader(
            "Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            model = load_model(modelName)
            predictions = predict_model(estimator=model, data=data).rename(
                columns={'Label': 'Charges'})
            print(predictions)
            st.write(predictions)


if __name__ == '__main__':
    run()
