import matplotlib.pyplot as plt
import seaborn as sns


def compare_actual_predicted_price (y_test, test_data_prediction):
    y_test = list(y_test)
    plt.plot(y_test, color='blue', label='Actual Value')
    plt.plot(test_data_prediction, color='red', label='Predicted Value')
    plt.title('Actual Price vs Predicted Price')
    plt.xlabel('Number of values')
    plt.ylabel('GLD Price')
    plt.legend()
    plt.show()