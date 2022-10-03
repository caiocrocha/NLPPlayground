# TextAnalysis

## An automatic writing skill classifier
This repo was developed for an internship selection project of CAEd UFJF. An interactive demo of the app is found at https://share.streamlit.io/caiocrocha/textanalysis/main/app.py. 

The notebooks [classification.ipynb](https://github.com/caiocrocha/TextAnalysis/blob/main/classification.ipynb) and [operations.ipynb](https://github.com/caiocrocha/TextAnalysis/blob/main/operations.ipynb) contain the text classification and text operations experiments. 

If you wish to run the app locally, the required packages listed in `requirements.txt` must be installed, according to the following instructions. 

## Instructions
Before following the installation instructions, having a virtual env set-up is recommended. Then, you can run the following commands: 
```
pip install -r requirements.txt
pip install streamlit
streamlit run app.py
```

Jupyter notebook is required in case you want to visualize the notebooks.
```
pip install notebook
jupyter notebook
```

To preprocess your own dataset, run the script `dataset_to_csv.py` in the `src` folder.
```
python3 dataset_to_csv.py dataset/
```

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

Para processar o seu próprio dataset, rode o script `dataset_to_csv.py` na pasta `src`.
```
python3 dataset_to_csv.py dataset/
```