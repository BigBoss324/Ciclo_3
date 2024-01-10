import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures,OrdinalEncoder,LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, r_regression
import warnings
import os
import seaborn as sns
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from math import sin, cos, pi


warnings.filterwarnings('ignore')

# Carga de datos
DATA_PATH = "C:/Users/Alexis Taha/Desktop/digital nao"
FILE_BIKERPRO = 'SeoulBikeData.csv'
bikerpro = pd.read_csv(os.path.join(DATA_PATH, FILE_BIKERPRO), encoding="ISO-8859-1")

# Limpieza y preprocesamiento de datos
clean_columns = [x.lower().replace("(°c)", '').replace("(%)", '').replace(" (m/s)", '').replace(" (10m)", '').replace(" (mj/m2)", '').replace("(mm)", '').replace(" (cm)", '').replace(" ", '_') for x in bikerpro.columns]
bikerpro.columns = clean_columns
bikerpro['date'] = pd.to_datetime(bikerpro['date'], format='%d/%m/%Y')

# Transformaciones de variables
bikerpro['hour'] = bikerpro['hour'].astype('category')
bikerpro['weekday'] = bikerpro['date'].dt.weekday.astype('category')
bikerpro['month'] = bikerpro['date'].dt.month.astype('category')
bikerpro['is_weekend'] = bikerpro['weekday'].isin([5, 6]).astype(int)
bikerpro['holiday'] = bikerpro['holiday'].map({'No Holiday': 0, 'Holiday': 1})
bikerpro['functioning_day'] = bikerpro['functioning_day'].map({'No': 0, 'Yes': 1})

# Definición de columnas para el modelado
weather_cols = ['temperature', 'humidity', 'wind_speed', 'visibility', 'solar_radiation', 'rainfall', 'snowfall']  # Excluyendo 'dew_point_temperature'
categorical_cols = ['seasons', 'hour', 'weekday']  # Añadiendo 'weekday'
binary_cols = ['holiday', 'functioning_day', 'is_weekend']  # Añadiendo 'functional_day'

# Preprocesamiento y división de los datos
X = bikerpro[weather_cols + categorical_cols + binary_cols]
y = bikerpro['rented_bike_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)



class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.lambdas_ = None

    def fit(self, X, y=None):
        self.lambdas_ = {}
        for col in self.columns:
            _, self.lambdas_[col] = stats.yeojohnson(X[col])
        return self

    def transform(self, X, y=None):
        X_trans = X.copy()
        for col in self.columns:
            X_trans[col] = stats.yeojohnson(X[col], lmbda=self.lambdas_[col])
        return X_trans


cols_to_transform = ['temperature', 'humidity', 'wind_speed', 'visibility','solar_radiation', 'rainfall', 'snowfall']
  

# Definición de los pipelines
numerical_pipe = Pipeline([
    ('yeojohnson', YeoJohnsonTransformer(columns=cols_to_transform)),
    ('standar_scaler', MinMaxScaler()),
    ('select_best', SelectKBest(mutual_info_regression, k=4)),
    ('polynomial', PolynomialFeatures(degree=2))
])

categorical_pipe = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

pre_processor = ColumnTransformer([
    ('numerical', numerical_pipe, weather_cols),
    ('categorical', categorical_pipe, categorical_cols),
], remainder='passthrough')

# Pipeline para Random Forest
pipe_rf = Pipeline([
    ('transform', pre_processor),
    ('model', RandomForestRegressor(random_state=0))
])


# Pipeline para Ridge
pipe_ridge = Pipeline([
    ('transform', pre_processor),
    ('model', Ridge(random_state=0))
])


# Pipeline para K-NN
pipe_knn = Pipeline([
    ('transform', pre_processor),
    ('model', KNeighborsRegressor())
])

# Parámetros para K-NN
param_grid_knn = {
    'model__n_neighbors': [20],
    'model__weights': ['distance'],
    'model__metric': ['manhattan']
}



# Parámetros para Random Forest
param_dist_rf = {
    'model__n_estimators': [100],
    'model__max_depth': [None],
    'model__min_samples_split': [2],
    'model__min_samples_leaf': [1],
    'model__max_features': [None],
    'model__min_impurity_decrease': [0.01],
    'model__bootstrap' : [True]
}

# Parámetros para Ridge
param_grid_ridge = {
    'model__alpha': [1]
}


# Ajuste y evaluación de cada modelo
cv_rf = random_search_rf = RandomizedSearchCV(
    pipe_rf,
    param_distributions=param_dist_rf,
    n_iter=100,    # Número de iteraciones de búsqueda
    cv=5,          # Validación cruzada de 5 pliegues
    scoring='neg_mean_squared_error',  # Puedes cambiar esto según tu métrica objetivo
    random_state=42,
    n_jobs=-1      # Usa todos los núcleos disponibles
).fit(X_train, y_train)

cv_ridge = RandomizedSearchCV(pipe_ridge, param_grid_ridge, n_jobs=-1, scoring='neg_mean_squared_error', cv=5).fit(X_train, y_train)
cv_knn = RandomizedSearchCV(pipe_knn, param_grid_knn, n_jobs=-1, scoring='neg_mean_squared_error', cv=5).fit(X_train, y_train)


# Evaluación de los modelos en el conjunto de prueba y de entrenamiento
error_rf_train = mean_squared_error(y_train, cv_rf.predict(X_train), squared=False)
error_ridge_train = mean_squared_error(y_train, cv_ridge.predict(X_train), squared=False)
error_knn_train = mean_squared_error(y_train, cv_knn.predict(X_train), squared=False)

# Imprimir resultados de RMSE para cada modelo en el conjunto de prueba
print("Evaluación de Modelos en el Conjunto de Entrenamiento")
print(f"Random Forest - RMSE: {error_rf_train.round(2)}")
print(f"Ridge - RMSE: {error_ridge_train.round(2)}")
print(f"K-NN - RMSE: {error_knn_train.round(2)}")


# Evaluación de los modelos en el conjunto de prueba
error_rf_test = mean_squared_error(y_test, cv_rf.predict(X_test), squared=False)
error_ridge_test = mean_squared_error(y_test, cv_ridge.predict(X_test), squared=False)
error_knn_test = mean_squared_error(y_test, cv_knn.predict(X_test), squared=False)

# Imprimir resultados de RMSE para cada modelo en el conjunto de prueba
print("Evaluación de Modelos en el Conjunto de Prueba")
print(f"Random Forest - RMSE: {error_rf_test.round(2)}")
print(f"Ridge - RMSE: {error_ridge_test.round(2)}")
print(f"K-NN - RMSE: {error_knn_test.round(2)}")


# Seleccionar el modelo con el menor error en el conjunto de prueba
errors = {'Random Forest': error_rf_test, 'Ridge': error_ridge_test, 'K-NN': error_knn_test}
best_model_name = min(errors, key=errors.get)
best_model = {'Random Forest': cv_rf, 'Ridge': cv_ridge, 'K-NN': cv_knn}[best_model_name]

# Asumiendo que best_model_rf es tu mejor modelo Random Forest
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Crear DataFrames para comparar los valores reales y predichos
train_comparison = pd.DataFrame({'Real': y_train, 'Predicción': y_pred_train, 'Error': y_train - y_pred_train})
test_comparison = pd.DataFrame({'Real': y_test, 'Predicción': y_pred_test, 'Error': y_test - y_pred_test})

# Puedes agrupar los datos por una característica específica para ver si hay patrones
error_by_hour = train_comparison.groupby(bikerpro['hour'])['Error'].mean()
plt.figure(figsize=(10, 5))
error_by_hour.plot(kind='bar')
plt.xlabel('Hora del Día')
plt.ylabel('Error Medio')
plt.title('Error Medio por Hora del Día')
plt.show()



# Imprimir el modelo con el mejor rendimiento
print(f"\nModelo con mejor rendimiento: {best_model_name} con RMSE: {errors[best_model_name].round(2)}")

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