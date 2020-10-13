docker build -t pycaret-streamlit:v1 .
docker run --name pycaret_streamlit_v1 -d -p 8501:8501 pycaret-streamlit:v1
