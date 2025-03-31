import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.randint(50, 1400, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

def train_model():
  df = generate_house_data(n_samples=100)
  x = df[['size']]
  y = df['price']
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

  model = LinearRegression()
  model.fit(x_train, y_train)
  return model

def main():
  st.title("Simple Linear Regression House Prediction App")
  st.write("Put in your house size to know its proce")
  model = train_model()
  size = st.number_input('House Size', min_value = 500, max_value = 2000, value = 1500)
  if st.button('Predict price'):
      predicted_price = model.predict([[size]])
      st.success(f"Estimated price: ${predicted_price[0]:,.2f}")
      df = generate_house_data()
      fig = px.scatter(df, x='size', y='price', title='House Size vs Price')
      fig.add_scatter(x=[size], y=[predicted_price[0]], mode='markers',  marker = dict(size = 15, color = 'red', name='Prediction'))
      st.plotly_chart(fig)

if __name__ == '__main__':
  main()
