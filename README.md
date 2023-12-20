<h2 align="center"> Drugs Classification </h2>

Notes (not so important): I made this in wsl ubuntu. So, if you want to run the codes (drugs-classification.ipynb, train.py) again make sure don't forget to change the paths.

<!-- ABOUT THE PROJECT -->

## About The Project

This project is for MLZoomcap Capstone 01. Drugs classification is about classifing drugs based on patient condition. There are 5 types of drugs: 'DrugY' 'drugC' 'drugX' 'drugA' 'drugB'. So, we will use features from dataset to classifing the drugs. To do that we can use logistic regression and tree based model. For this project, I use from Kaggle:

[https://www.kaggle.com/datasets/prathamtripathi/drug-classification](https://www.kaggle.com/datasets/prathamtripathi/drug-classification)

Features in the dataset are: 'Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'. with 'Drug' becomes the target. You can find the dataset in repository here too at `data/drug200.csv`

The purpose of this project are to classification drugs based on the patient conditions (features). This will be useful in the health sector.

## About Files

- `drugs-classification.ipynb` = EDA and Model Training (select Model).
- `train.py` = script for final model
- `predict.py` = script for loading the model and serving it via web service
- `test.py` = for testing

## How to run the project

Download or clone the project `https://github.com/rahmaha/drugs-classification.git`

### Locally

1. First, you need to have docker and pipenv on your laptop.
2. Build the docker image

`docker build -t drugs-classification .` You can change the name of the docker image as you want.

3. Run the docker image

`docker run -it -p 9696:9696 drugs-classification:latest .`

4. Run pipenv install
5. Run pipenv shell
6. Uncomment url (localhost) in the `test.py` and comment the other url (elasticbeanstalk)
7. Open new terminal, run `test.py` script.

### Cloud

For cloud, I using aws EBS. To run the project, you can:

1. Run pipenv install
2. Run pipenv shell
3. (Optional) you can change test.py script tp submit request.

```import requests

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
```

| SEX  |   BP   | CHOLESTEROL |
| :--: | :----: | :---------: |
|  F   |  HIGH  |    HIGH     |
|  M   |  LOW   |   NORMAL    |
| Good | NORMAL |     VS1     |
|      |        |             |

3. Open new terminal, run `test.py` script.
