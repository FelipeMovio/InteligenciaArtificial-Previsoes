# Criar um sistema de forma automatica que define se o cliente é seguro ou nao 
    # Score de credito do cliente 
    # Boa
    # Ok
    # Ruim

# Passo 1 : importar base de dados 

import pandas as pd

tabela = pd.read_csv("./clientes.csv")
