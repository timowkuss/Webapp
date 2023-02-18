import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler,  LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

st.write("""
## Это приложение **сегментирует** клиентов магазина!
""")
st.sidebar.header("Введите параметры")

customers = st.sidebar.text_input('Имя', '', key='customers')
age = st.sidebar.slider('Возраст', 18, 55, 35, key='age')
brand = st.sidebar.text_input('Бренд машины', '', key='brand')
model = st.sidebar.text_input('Модель машины', '', key='model')
year_issue = st.sidebar.slider('Год выпуска машины', 2009, 2022, 2015, key='year_issue')
price = st.sidebar.slider('Цена', 5000000, 40000000, 25000000, 50000, key='price')
category = st.sidebar.selectbox('Выберите категорию машины',('A', 'B', 'C'), key='category')
credit = st.sidebar.radio("Машина куплена в кредит?",('Yes', 'No'), key='credit')
tradein = st.sidebar.radio("Покупали через Trade-in?",('Yes', 'No'), key='tradein')
data = {'Customers': customers,
        'Age': age,
        'Brand': brand,
        'Model': model,
        'Year_issue': year_issue,
        'Price': price,
        'Category': category,
        'Credit': credit,
        'Trade-in': tradein}
df_orig = pd.DataFrame([data], index=[0])
X_orig = pd.read_excel('diplom.xlsx', sheet_name = "customers")
X = X_orig.copy()
X_data = X_orig.copy()
df_data = df_orig.copy()

# Data preprocessing
del_columns = ['Customers', 'Brand',  'Model', 'Trade-in', 'Category']
X_data = X_data.drop(del_columns, axis = 1)
X_orig = X_orig.drop(del_columns, axis = 1)
df_orig = df_orig.drop(del_columns, axis = 1)
res = pd.concat([X_orig, df_orig], ignore_index=True)
# LabelEncoder
LE = preprocessing.LabelEncoder()
X_data["Credit"] = LE.fit_transform(X_data["Credit"])
res["Credit"] = LE.fit_transform(res["Credit"])
# StandardScaler
scaler = StandardScaler()
X_data_scaled = scaler.fit_transform(X_data)
X_data_scaled = pd.DataFrame(X_data_scaled)
X_scaled = scaler.fit_transform(res)
X_scaled = pd.DataFrame(X_scaled)

# Data PCA
pca = PCA(n_components = 2)
X_data_principal = pca.fit_transform(X_data_scaled)
X_data_principal = pd.DataFrame(X_data_principal)
X_data_principal.columns = ['P1', 'P2']
X_principal = pca.fit_transform(X_scaled)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
st.subheader('Введенные параметры клиента')
df_principal = X_principal.iloc[-1:]

# K-Means Algotirgm 
kmeans = KMeans(n_clusters=4, init = "k-means++", random_state = 42)
kmeans.fit(X_data_principal)
kmeans_labels = kmeans.labels_
X['Cluster'] = kmeans.labels_
X_data_principal['Cluster'] = kmeans.labels_

# Prediction
name = ['new']
prediction = kmeans.predict(df_principal)
df_data['Cluster'] = prediction
df_principal['Cluster'] = prediction
df_data['name'] = name
st.write(df_data)
#st.dataframe(df_data.drop(columns='name'))
#st.dataframe(X)
fig = plt.figure()
sns.scatterplot(x = 'P1', y = 'P2', hue = 'Cluster',  data = X_data_principal, palette='viridis')
sns.scatterplot(x = 'P1', y = 'P2', hue = 'Cluster',  data = df_principal, s=120, palette="viridis")
st.pyplot(fig)

st.pyplot(sns.pairplot(X, hue='Cluster', palette = ['red', 'green', 'blue', 'purple'], y_vars=["Price", "Age", "Year_issue"], x_vars=["Price", "Age", "Year_issue", 'Credit']))


fig1 = plt.figure()
sns.scatterplot(x = 'Age', y = 'Price', hue = 'Cluster',  data = X, palette='viridis')
sns.scatterplot(x = 'Age', y = 'Price', hue = 'Cluster',  data = df_data, s=120, palette="viridis")
st.pyplot(fig1)






