import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import autograd.numpy as np_   # Thinly-wrapped version of Numpy
from autograd import grad

df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
df = df.dropna() # Dropa linhas com valores nulos
df = df[df["gender"] != "Other"] # Dropa linha onde gender = other
df.replace({"stroke": 0}, -1, inplace=True) # Substitui 0 por -1
df["stroke"].value_counts()

df_sem_stroke = df.drop(columns=["stroke", "id"])
df_sem_stroke = pd.get_dummies(df_sem_stroke, drop_first=True)

def datasets(df_sem_stroke, df):
    X_train, X_test, y_train, y_test = train_test_split(df_sem_stroke, df["stroke"], train_size=0.5) # Divide o dataset em treino e teste

    # Transforma o df em um numpy array
    X_train = X_train.to_numpy(dtype=float).T # Transpõe o array para que cada coluna seja um ponto
    X_test = X_test.to_numpy(dtype=float).T # Transpõe o array para que cada coluna seja um ponto
    y_train = y_train.to_numpy(dtype=float).T # Transpõe o array para que cada coluna seja um ponto
    y_test = y_test.to_numpy(dtype=float).T # Transpõe o array para que cada coluna seja um ponto
    return X_train, X_test, y_train, y_test

def loss( parametros ): # Essa função calcula o erro médio quadrático
    w, b, pontos, val = parametros
    est = w.T @ pontos + b
    mse = np_.mean( (est - val)**2)
    return mse

def accuracy(y_test, y_est):
    return np.mean(np.sign(y_test)==np.sign(y_est))

def calcula_acuracia(df_sem_stroke, df):
    X_train, X_test, y_train, y_test = datasets(df_sem_stroke, df)

    g = grad(loss)

    w = np.random.randn(15 ,1) # Vetor de pesos
    w_ = w
    b = 0.0 # Viés / bias
    alpha = 10**-5

    for n in range(10000):
        grad_ = g( (w, b, X_train, y_train) )
        w -= alpha*grad_[0]
        b -= alpha*grad_[1]

    y_est = w.T @ X_test + b # Estimativa
    acc = accuracy(y_test, y_est) # Acurácia
    print(f"Accuracy -> {acc}")
    print(w_)

calcula_acuracia(df_sem_stroke, df)