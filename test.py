import requests

# url = "http://localhost:9696/predict"  # to run locally using docker
host = "coba-capstone1-env.eba-kftbizs5.ap-southeast-1.elasticbeanstalk.com."
url =  f"http://{host}/predict"

patient = {
    'Age': 30,
    'Sex': 'M',
    'BP': 'HIGH',
    'Cholesterol': 'NORMAL',
    'Na_to_K': 16.76
}

response = requests.post(url, json=patient).json()
print('Predicted drug: ', response.get('drug'))
