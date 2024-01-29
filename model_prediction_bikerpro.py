# lets import train test split for splitting the data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import  MinMaxScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import warnings
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import pickle
import seaborn as sns
from sklearn.compose import ColumnTransformer


warnings.filterwarnings('ignore')

# Carga de datos
DATA_PATH = "C:/Users/Alexis Taha/Desktop/digital nao"
FILE_BIKERPRO = 'SeoulBikeData.csv'
bikerpro = pd.read_csv(os.path.join(DATA_PATH, FILE_BIKERPRO), encoding="ISO-8859-1")


bikerpro['Date'] = pd.to_datetime(bikerpro['Date'], format='%d/%m/%Y')

bikerpro['Month'] = bikerpro['Date'].dt.month.astype('category')
bikerpro['Day'] = bikerpro['Date'].dt.day_name()    
bikerpro['Weekdays_or_weekend'] = bikerpro['Day'].apply(lambda x: 1 if x=='Saturday' or x=='Sunday' else 0)

bikerpro.drop(['Date',"Day"],axis=1 ,inplace=True)

df = bikerpro.copy()

df.drop(['Humidity(%)','Visibility (10m)'],inplace=True,axis=1)

df['Spring'] = np.where(df['Seasons'] == 'Spring', 1, 0)
df['Summer'] = np.where(df['Seasons'] == 'Summer', 1, 0)
df['Autumn'] = np.where(df['Seasons'] == 'Autumn', 1, 0)
df['Winter'] = np.where(df['Seasons'] == 'Winter', 1, 0)

df.drop(columns=['Seasons'],axis=1,inplace=True)
df['Holiday'] = df['Holiday'].map({'No Holiday':0, 'Holiday':1})
df['Functioning Day'] = df['Functioning Day'].map({'Yes':1, 'No':0})

y = np.sqrt(df['Rented Bike Count'])

X = df.drop('Rented Bike Count',axis=1)

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Definición de los pipelines
numerical_pipe = Pipeline([  # Uso de PowerTransformer
    ('min_max_scaler', MinMaxScaler())
])

categorical_pipe = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, numerical_cols),
    ('categorical', categorical_pipe, categorical_cols),
], remainder='passthrough')

# Pipeline para Random Forest
pipe_rf = Pipeline([
    ('transform', numerical_pipe),
    ('model', RandomForestRegressor(random_state=0))
])


# Pipeline para Ridge
pipe_ridge = Pipeline([
    ('transform', numerical_pipe),
    ('model', Ridge(random_state=0))
])


# Pipeline para K-NN
pipe_knn = Pipeline([
    ('transform', numerical_pipe),
    ('model', KNeighborsRegressor())
])


# Parámetros para K-NN
param_grid_knn = {
    'model__n_neighbors': [5, 10, 15, 20, 25],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan', 'minkowski']
}


# Parámetros para Random Forest
param_dist_rf = {
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [5, 10, 15, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Parámetros para Ridge
param_grid_ridge = {
    'model__alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
}

tscv = TimeSeriesSplit(n_splits=4, test_size=2)

cv_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_dist_rf,  # Uso de 'param_grid'
    cv=tscv,                   # TimeSeriesSplit como estrategia de validación cruzada
    scoring='neg_mean_squared_error',
    n_jobs=-1                  # Usa todos los núcleos disponibles
)

cv_rf.fit(X_train, y_train)


cv_ridge = GridSearchCV(
estimator=pipe_ridge,
param_grid=param_grid_ridge,
cv=tscv, # TimeSeriesSplit como estrategia de validación cruzada
scoring='neg_mean_squared_error',
n_jobs=-1 # Usa todos los núcleos disponibles
)

cv_ridge.fit(X_train, y_train)


cv_knn = GridSearchCV(
estimator=pipe_knn,
param_grid=param_grid_knn,
cv=tscv, # TimeSeriesSplit como estrategia de validación cruzada
scoring='neg_mean_squared_error',
n_jobs=-1 # Usa todos los núcleos disponibles
)

cv_knn.fit(X_train, y_train)

print("------------------------")

# Mejores parámetros para Random Forest
print("Mejores parámetros para Random Forest:")
print(cv_rf.best_params_)

# Mejores parámetros para Ridge
print("\nMejores parámetros para Ridge:")
print(cv_ridge.best_params_)

# Mejores parámetros para K-NN
print("\nMejores parámetros para K-NN:")
print(cv_knn.best_params_)

print("-------------------------")
print(" ")

# Evaluación de los modelos en el conjunto de prueba y de entrenamiento
error_rf_train = mean_squared_error(y_train, cv_rf.predict(X_train), squared=False)
error_ridge_train = mean_squared_error(y_train, cv_ridge.predict(X_train), squared=False)
error_knn_train = mean_squared_error(y_train, cv_knn.predict(X_train), squared=False)

print("-------------------------")
# Imprimir resultados de RMSE para cada modelo en el conjunto de prueba
print("Evaluación de Modelos en el Conjunto de Entrenamiento")
print(f"Random Forest - RMSE: {error_rf_train}")
print(f"Ridge - RMSE: {error_ridge_train}")
print(f"K-NN - RMSE: {error_knn_train}")
print("-------------------------")
print(" ")

# Evaluación de los modelos en el conjunto de prueba
error_rf_test = mean_squared_error(y_test, cv_rf.predict(X_test), squared=False)
error_ridge_test = mean_squared_error(y_test, cv_ridge.predict(X_test), squared=False)
error_knn_test = mean_squared_error(y_test, cv_knn.predict(X_test), squared=False)

print("-------------------------")
# Imprimir resultados de RMSE para cada modelo en el conjunto de prueba
print("Evaluación de Modelos en el Conjunto de Prueba")
print(f"Random Forest - RMSE: {error_rf_test}")
print(f"Ridge - RMSE: {error_ridge_test}")
print(f"K-NN - RMSE: {error_knn_test}")
print("-------------------------")
print(" ")

# Seleccionar el modelo con el menor error en el conjunto de prueba
errors = {'Random Forest': error_rf_test, 'Ridge': error_ridge_test, 'K-NN': error_knn_test}
best_model_name = min(errors, key=errors.get)
best_model = {'Random Forest': cv_rf, 'Ridge': cv_ridge, 'K-NN': cv_knn}[best_model_name]

# Hacer predicciones
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Crear DataFrames para comparar los valores reales y predichos
train_comparison = pd.DataFrame({'Real': y_train, 'Predicción': y_pred_train})
train_comparison['Error'] = train_comparison['Real'] - train_comparison['Predicción']

test_comparison = pd.DataFrame({'Real': y_test, 'Predicción': y_pred_test})
test_comparison['Error'] = test_comparison['Real'] - test_comparison['Predicción']


# Imprimir el modelo con el mejor rendimiento
print(f"\nModelo con mejor rendimiento: {best_model_name} con RMSE: {errors[best_model_name]}")

# Configuración de estilo de gráficos
sns.set_style("whitegrid")

# Configuración de estilo de gráficos
sns.set_style("whitegrid")

# Función para graficar comparaciones
def plot_comparations(y_true, y_pred, set_name, model_name):
    plt.figure(figsize=(10, 6))
    # Limitar la cantidad de puntos a graficar para claridad
    sample_size = min(1000, len(y_true))
    indices = np.linspace(0, len(y_true) - 1, sample_size).astype(int)
    
    # Si y_true es una Serie de pandas, usa iloc para acceder por posición
    if isinstance(y_true, pd.Series):
        y_true_sample = y_true.iloc[indices]
    else:
        y_true_sample = y_true[indices]
    
    plt.plot(indices, y_true_sample, label='Actual', color='blue', alpha=0.6, linewidth=2)
    plt.plot(indices, y_pred[indices], label='Predicción', color='cyan', alpha=0.7, linewidth=2)
    
    plt.title(f'Comparación entre Valores Reales y Predicciones - {set_name}')
    plt.xlabel('Muestras')
    plt.ylabel('Rented Bike Count')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()  # Ajusta la subtrama para que la leyenda no se superponga
    plt.savefig(f'comparative_actual_model_{set_name.lower()}_set.png')
    plt.show()


with open('model_prediction_bikerpro.pkl', 'wb') as file:
    pickle.dump(best_model, file)


# Graficar comparaciones para el conjunto de entrenamiento
plot_comparations(y_train, best_model.predict(X_train), 'Train', best_model_name)

# Graficar comparaciones para el conjunto de prueba
plot_comparations(y_test, best_model.predict(X_test), 'Test', best_model_name)
