from visualizator import compare_actual_predicted_price
from App import App as AppCustom

CSV_FILE_PATH = 'datasets/gld_price_data.csv'


def gold_prediction_custom():
    app = AppCustom(CSV_FILE_PATH)
    app.standardize_data()
    app.fit()
    app.predict()
    app.error_score()

    y_test = app.get_y_test()
    test_prediction = app.get_test_prediction()
    compare_actual_predicted_price(y_test, test_prediction)


gold_prediction_custom()




