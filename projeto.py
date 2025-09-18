# Criar um sistema de forma automatica que define se o cliente é seguro ou nao 
    # Score de credito do cliente 
    # Boa - Good
    # Ok - Standard
    # Ruim - Poor

# Passo 1 : importar base de dados 

import pandas as pd

tabela = pd.read_csv("./clientes.csv")

# Passo 2 : tratamento de dados / prepara a base de dados para a IA

print(tabela.info())
# Ver informações da base de dados

# LabelEncoder
# Transformando valores na base de dados que sao objetos/textos em numeros 
from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

tabela["profissao"] = codificador.fit_transform(tabela["profissao"])

tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])

tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])


# separar as informações da base de dados para a IA 
    # separar em X e Y 
    # Y -> quem eu quero prever (coluna score_credito)
    # X -> as outras colunas 

y = tabela["score_credito"]        
x = tabela.drop(columns = ["score_credito","id_cliente"])

# separa em dados de treino e de teste 
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3)


# Passo 3 : Criar o modelo de IA -> prever a nota de credito 

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvoredecisao = RandomForestClassifier()

modelo_knn = KNeighborsClassifier()

modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)


# Passo 4 : Escolher o melhor modelo 

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

previsa_knn = modelo_knn.predict(x_teste)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_teste, previsao_arvoredecisao))
print(accuracy_score(y_teste, previsa_knn))

# Melhor modelo é o de arvore de decisao 