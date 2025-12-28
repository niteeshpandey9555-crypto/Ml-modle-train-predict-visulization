import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'area': [500, 800, 1200, 1700, 2100],
        'age': [6, 8, 11, 14, 17],
            'bedroom': [3, 5, 7, 9, 11],
                'price': [50000, 70000, 90000, 110000, 130000]
}
df = pd.DataFrame(data)
model = LinearRegression()
X = df[['area', 'age', 'bedroom']]
y = df['price']
model.fit(X, y)

plt.scatter(df.area,df.price, color='red',marker='*',s=110)
plt.plot(df.area, model.predict(X), color='blue')
plt.grid(color='green',linestyle='--' )
plt.title('house area pice ')
plt.xlabel('area')
plt.ylabel('price')


print(model.predict([[1500, 10, 5]]))
