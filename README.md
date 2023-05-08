# cars-api

Simple flask implementation for deploying machine learning model as a REST API.

## Install Requrements

````console
pip install -r requrements.
````

## Model

You can download the model weights [here](https://github.com/hfsykr/cars-classification/blob/main/output/mobilenet_v3_l/weights.pt) and move it to 'model/mobilenet_v3_l' folder [(here)](model/mobilenet_v3_l).

## Local/Development Run

````console
gunicorn --bind 127.0.0.1 wsgi:app
````

## API Call Example

Request:

````console
curl -X POST http://127.0.0.1:8000/predict -F image=@path/to/image/cars_image.jpg
````

Response:

````console
{
    "class_name": "Tesla Model S Sedan 2012",
    "label": 184
}
````

