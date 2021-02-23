# from logging import debug
from pycaret.regression import predict_model, load_model
# from pycaret.datasets import get_data
import streamlit as st
import sys
import pandas as pd
from datetime import datetime
import time
import os
import base64
import requests
# import fnmatch
from pathlib import Path
import math
import zipfile
import glob

IMG_PATH = 'public/static/img/'
MODEL_PATH = 'volume/models/'
AUTOML_API_URL = 'http://localhost:5000'
Features = {
    'insurance': ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
    'diamond': ['Carat', 'Weight', 'Cut', 'Color',
                'Clarity', 'Polish', 'Symmetry', 'Report'],
    'iris': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
}
LabelEncoded = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


def predict(model, input_df):
    try:
        predictions_df = predict_model(estimator=model, data=input_df)
        predictions = predictions_df['Label'][0]
        return predictions
    except KeyError:
        st.warning("予測したいTargetはモデルと違って、正しい「Target」を選んでください。")


def find(pattern, path):
    result = []
    # for root, dirs, files in os.walk(path):
    #     for name in files:
    #         if fnmatch.fnmatch(name, pattern):
    #             result.append(os.path.splitext(name)[0])
    for name in glob.iglob(path + '**/' + pattern, recursive=True):
        result.append(os.path.basename(os.path.splitext(name)[0]))
    return result


def trainModel(dType, dataset, target):
    if st.button("モデル新規作成"):
        url = AUTOML_API_URL + '/automl'
        myobj = {'type': dType, 'target': target, 'dataset': dataset}
        chunk_size = 1024
        'Waiting for training model...'
        r = requests.post(url, data=myobj, stream=True)
        if r.status_code == 200:
            content_size = int(r.headers['content-length'])
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d%H%M%S")
            fileName = dataset + '_' + dt_string + '.zip'
            with open(MODEL_PATH + dataset + '/' + fileName, "wb") as file:
                # Add a placeholder
                latest_iteration = st.empty()
                'Downloading the model...'
                bar = st.progress(0)
                size = 0
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        size = len(chunk) + size
                        # Update the progress bar with each iteration.
                        latest_iteration.text(
                            f'Iteration {int(size/content_size)*100}')
                        bar.progress(int(size / content_size) * 100)
                        time.sleep(0.1)
            with zipfile.ZipFile(MODEL_PATH + dataset + '/' + fileName ) as existing_zip:
                existing_zip.extractall(MODEL_PATH + dataset + '/')
            st.success(
                '{}のモデル{}の作成が完了しました'.format(target, dataset))
        else:
            st.error("{}のモデル{}の作成に失敗しました".format(target, dataset))


def crtDataset():
    uploaded_file = st.file_uploader("学習データのcsvファイルを選んでください。", type=["csv"])
    if uploaded_file is not None:
        if st.button("データセット作成"):
            url = AUTOML_API_URL + '/dataset'
            myobj = {'fileupload': uploaded_file}
            r = requests.post(url, files=myobj)
            res = r.json()
            st.info(res["result"])


def delDataset(toDeleteDataset):
    if st.button("当該データセット削除"):
        url = AUTOML_API_URL + '/dataset'
        myobj = {'name': toDeleteDataset}
        r = requests.delete(url, data=myobj)
        res = r.json()
        st.info(res["result"])


def getDatasets():
    url = AUTOML_API_URL + '/dataset'
    r = requests.get(url)
    res = r.json()
    return res["data"]


def getDatasetHeader(name):
    url = AUTOML_API_URL + '/dataset'
    payload = {'name': name}
    r = requests.get(url, params=payload)
    res = r.json()
    df = pd.DataFrame(res["data"])
    return list(df)


def delModel(toDeleteModel, dataset):
    if st.button("当該モデル削除"):
        try:
            os.remove(MODEL_PATH + dataset + '/' + toDeleteModel + '.pkl')
            st.info("{}モデル削除しました。".format(toDeleteModel))
        except FileNotFoundError:
            st.warning("{}モデルが見つかりません。".format(toDeleteModel))


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">予測結果のダウンロード</a>'
    return href


def run():
    from PIL import Image
    image = Image.open(IMG_PATH + 'logoMilizePycaret.png')
    image_milize = Image.open(IMG_PATH + 'milize_logo.png')
    irisImage = Image.open(IMG_PATH + 'iris-machinelearning.jpg')
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # body {color: white;}
            .css-1aumxhk {
                background-image: linear-gradient(#008a07,#008a07);
                color: white;
            }
            .css-1ilyi7m {
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
            .uploadedFileName{
                color: white;
            }
            .stNumberInput div.input-container div.controls .control{
                background-color:#008a07;
            }
            .ReactVirtualized__Grid{
                color: white;
            }
            h1{
                color: white
            }
            .st-au{
                border: 0;
                background-image: linear-gradient(#008a07,#008a07);
            }
            .stSelectbox label{
                color: white;
            }
            .step-down{
                color: white;
                background-color:#1e3216;
            }
            .step-up{
                color: white;
                background-color:#1e3216;
            }
            .st-bp{
                color: white;
                background-color:#1e3216;
            }
            .st-bs {
                color: white;
            }
            .st-fg {
                color: white;
            }
            .st-dc{
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
            .st-ef {
                color:white;
            }
            .css-2trqyj {
                color:white;
                background-color:#1e3216;
                border: 0;
            }
            .css-odwihv {
                color:#008a07;
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
    st.sidebar.info('Copyright © Milize 2020~2021')

    st.title("AutoML予測アプリ")
    dType = st.selectbox('Type', ['Regression', 'Classification'])
    ds = getDatasets()
    if(dType == 'Regression'):
        ds.append('insurance')
    elif(dType == 'Classification'):
        ds.append('iris')
    dataset = st.selectbox('Dataset', ds)
    if dataset == 'insurance':
        targetVal = 'charges'
    elif dataset == 'iris':
        targetVal = 'species'
    else:
        targetVal = getDatasetHeader(dataset)
    crtDataset()
    if len(ds) > 0:
        delDataset(dataset)
        Path(MODEL_PATH + dataset).mkdir(parents=True, exist_ok=True)
        models = find('*.pkl', MODEL_PATH + dataset + '/')
        if dataset == 'insurance' or dataset == 'iris':
            target = st.selectbox('Target', [targetVal])
        else:
            target = st.selectbox('Target', targetVal)
        if len(models) == 0:
            st.warning("モデルはありません。新規作成してください。")
            trainModel(dType, dataset, target)
        elif add_selectbox == 'Online':
            modelName = st.selectbox('Model', models)
            trainModel(dType, dataset, target)
            delModel(modelName, dataset)
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
                input_dict = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
            elif dataset == 'iris':
                sepal_length = st.number_input(
                    'sepal_length', min_value=1.0, max_value=100.0, value=5.1)
                sepal_width = st.number_input(
                    'sepal_width', min_value=1.0, max_value=50.0, value=3.5)
                petal_length = st.number_input(
                    'petal_length', min_value=1.0, max_value=100.0, value=1.4)
                petal_width = st.number_input(
                    'petal_width', min_value=0.0, max_value=50.0, value=0.2)
                input_dict = {'sepal_length': sepal_length, 'sepal_width': sepal_width, 'petal_length': petal_length, 'petal_width': petal_width}
            else:
                for idx, col in enumerate(targetVal):
                    if col != target:
                        locals()['v' + str(idx)] = st.number_input(col, min_value=0.0, value=0.0)
                        input_dict[col] = locals()['v' + str(idx)]
            output = ""
            input_df = pd.DataFrame([input_dict])

            model = load_model(MODEL_PATH + dataset + '/app/models/' + modelName + '/' + modelName)
            if st.button("予測する"):
                output = predict(model=model, input_df=input_df)
                if dataset == 'insurance':
                    output = '保険料予測は {}'.format(
                        str(math.ceil(output * 104.86)) + '円')
                elif dataset == 'iris':
                    output = 'This is ' + LabelEncoded[output]
                st.success(output)
                if dataset == 'iris':
                    st.image(irisImage, width=500)

        elif add_selectbox == 'Batch':
            modelName = sex = st.selectbox('Model', models)
            file_upload = st.file_uploader(
                "予測データのcsvファイルを選んでください。", type=["csv"])

            if file_upload is not None:
                print(file_upload)
                data = pd.read_csv(file_upload)
                model = load_model(MODEL_PATH + dataset + '/app/models/' + modelName + '/' + modelName)
                predictions = predict_model(estimator=model, data=data).rename(
                    columns={'Label': target})
                predictions[target].replace(
                    {0: LabelEncoded[0], 1: LabelEncoded[1], 2: LabelEncoded[2]}, inplace=True)

                st.write(predictions)
                if dataset == 'iris':
                    st.image(irisImage, width=500)

                st.markdown(get_table_download_link(predictions), unsafe_allow_html=True)


if __name__ == '__main__':
    args = sys.argv
    if(len(args) > 1):
        host = args[1]
        AUTOML_API_URL = AUTOML_API_URL.replace('localhost', host)
        print("host:", AUTOML_API_URL)
    run()
