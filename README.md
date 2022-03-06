# TextAnalysis

## Um classificador automático do nível de escrita de um texto

Este repositório foi desenvolvido com vista à prova prática de seleção para estágio no CAEd UFJF. Um site de apresentação da aplicação pode ser acessado pelo link https://share.streamlit.io/caiocrocha/textanalysis/main/app.py. 

Os notebooks [classification.ipynb](https://github.com/caiocrocha/TextAnalysis/blob/main/classification.ipynb) e [operations.ipynb](https://github.com/caiocrocha/TextAnalysis/blob/main/operations.ipynb) contém os experimentos realizados de classificação de texto e do impacto de operações de mutação na classificação. 

Caso deseje executar o aplicativo localmente, será necessário instalar os pacotes utilizados, conforme as instruções adiante. 

## Instruções
Antes de seguir as instruções de instalação, é recomendado utilizar um ambiente virtual no Python. Executar no terminal de comando os seguintes comandos: 
```
pip install -r requirements.txt
pip install streamlit
streamlit run app.py
```
Para visualizar os notebooks, é necessário instalar o jupyter notebook:
```
pip install notebook
jupyter notebook
```