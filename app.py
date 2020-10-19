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


def run():
    from PIL import Image
    image = Image.open(IMG_PATH+'logo.png')
    image_hospital = Image.open(IMG_PATH+'hospital.jpg')

    st.image(image, use_column_width=False)
    fileName = 'best_model_insurance_20201019182828'
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ('Training', "Online", "Batch"))

    st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.success('https://www.pycaret.org')

    st.sidebar.image(image_hospital)

    st.title("Insurance Charges Prediction App")
    if add_selectbox == 'Training':
        dataset = st.selectbox('Dataset', ['insurance', 'house'])
        targetVal = ''
        if dataset == 'insurance':
            targetVal = 'charges'
        elif dataset == 'house':
            targetVal = 'SalePrice'
        target = st.selectbox('Target', [targetVal])
        if st.button("AutoML"):
            url = 'http://127.0.0.1:5000/automl'
            myobj = {'target': targetVal, 'dataset': dataset}
            r = requests.post(url, data=myobj)
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d%H%M%S")
            fileName = 'best_model_'+dataset+'_'+dt_string+'.pkl'
            with open(fileName, "wb") as pkl:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        pkl.write(chunk)
            # data = get_data(dataset)
            # s1 = setup(data, target=targetVal, silent=True, html=False)
            # top3 = compare_models()
            # # finalize best model
            # best = finalize_model(top3)
            # now = datetime.now()
            # dt_string = now.strftime("%Y%m%d%H%M%S")
            # modelName = 'best_model_'+dataset+'_'+dt_string
            # save_model(best, modelName)
            # fileName = modelName + '.pkl'
            # 'Starting a long computation...'

            # # Add a placeholder
            # latest_iteration = st.empty()
            # bar = st.progress(0)

            # for i in range(100):
            #     # Update the progress bar with each iteration.
            #     latest_iteration.text(f'Iteration {i+1}')
            #     bar.progress(i + 1)
            #     time.sleep(0.1)

            # '...and now we\'re done!'
            st.success(
                'The model of predicting {} from {} has been finished'.format(target, dataset))
    models = find('*.pkl', './')
    print(models)
    if add_selectbox == 'Online':
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

    if add_selectbox == 'Batch':
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
