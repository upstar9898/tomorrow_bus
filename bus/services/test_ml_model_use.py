from .test_ml_model_load import ml_model1, ml_model2
from .test_ml_model_preprocess import preprocess

def predict(input_json):
    route_id = input_json.route_id
    station_id = input_json.station_id
    date_time = input_json.date_time
    x = preprocess(input_json)
    y = [ml_model1.predict(x), ml_model2.predict(x)]
    return y