import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

bike = pd.read_csv('bikes.csv')

bike.plot(kind='scatter', x='humidity', y='rentals')
plt.show()
depvar = 'rentals'
y = bike[depvar]
indepvar = list(bike.columns)
indepvar.remove(depvar)
x = bike[indepvar]
try:
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        random_state=1234)
    model = LinearRegression().fit(x_train, y_train)
    # print(model.score(x_test, y_test))
    # y_pred = model.predict(x_test)
    # from sklearn.metrics import mean_absolute_error
    # print(mean_absolute_error(y_test, y_pred))

    def predict_rentals(temperature, humidity, windspeed):
        input_data = pd.DataFrame({'temperature': [temperature], 'humidity': [humidity], 'windspeed': [windspeed]})
        prediction = model.predict(input_data)
        return round(prediction[0])

    # Get user input
    temperature = float(input("Enter the temperature: "))
    humidity = float(input("Enter the humidity: "))
    windspeed = float(input("Enter the windspeed: "))
    
    predicted_rentals = predict_rentals(temperature, humidity, windspeed)
    print(f"Predicted number of rentals: {predicted_rentals}")
except Exception as e:
    print(e)