from .test_ml_model_load import ml_model1, ml_model2, ml_model3
from .test_ml_model_preprocess import preprocess

def predict(input_json):
    route_id = input_json.routeId
    station_id = input_json.stationId
    date_time = input_json.dateTime
    x = preprocess(input_json)
    y = [ml_model1.predict(x), ml_model2.predict(x), ml_model3.predict(x)]
    return y