# import plotly.express as px
import pandas as pd
#
path = "C:\\Projects\\python\\currencyPriceEstimator\\data\\BNB_USD.csv"
df = pd.read_csv(path)
# df = px.data.stocks()
# fig = px.line(df, x="date", y=df.columns,
#               hover_data={"date": "|%B %d, %Y"},
#               title='custom tick labels')
# fig.update_xaxes(
#     dtick="M1",
#     tickformat="%b\n%Y")
# fig.show()


import matplotlib.pyplot as plt
import datetime
import numpy as np

# x = np.array([datetime.datetime(2013, 9, 28, i, 0) for i in range(24)])
# y = np.random.randint(100, size=x.shape)
plt.plot(df["Tarih"], df["Şimdi"])
# plt.plot(x,y)
plt.show()



