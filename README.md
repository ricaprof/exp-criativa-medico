# Medical Prescription AI

Sistema de **Inteligência Artificial para recomendação de medicamentos** baseado em dados de saúde.  
O objetivo do projeto é utilizar um **dataset real de medicamentos** para construir um modelo capaz de sugerir possíveis tratamentos com base em sintomas ou condições médicas.

>  **Aviso:** Este projeto possui finalidade **educacional e de pesquisa**. Não deve ser utilizado para diagnóstico ou prescrição médica real.

---

#  Objetivo

Este projeto tem como objetivo desenvolver um sistema de IA capaz de:

- Analisar **informações estruturadas sobre medicamentos**
- Identificar **possíveis usos terapêuticos**
- Sugerir **medicamentos candidatos para determinadas condições**
- Futuramente verificar **interações medicamentosas**
- Explorar o uso de **Machine Learning e NLP em dados de saúde**

---

#  Dataset

O projeto utiliza o dataset:

**Medicines Information Dataset (MID)**

Este dataset contém aproximadamente:

- **192.000 medicamentos**
- Informações como:
  - Nome do medicamento
  - Classe terapêutica
  - Composição
  - Benefícios
  - Efeitos colaterais
  - Instruções de uso
  - Classe química

 Fonte do dataset:  
https://data.mendeley.com/datasets/2vk5khfn6v/1

---

#  Arquitetura do Sistema

O sistema será dividido em três componentes principais:

## 1️⃣ Data Processing

Responsável por:

- Limpeza do dataset
- Normalização de textos
- Extração de features
- Transformação para formato utilizável pelo modelo

Tecnologias possíveis:

- Python
- Pandas
- NumPy
- Scikit-learn

---

## 2️⃣ Modelo de IA

Responsável por:

- Processar sintomas ou descrições de condições médicas
- Comparar com os dados dos medicamentos
- Retornar possíveis recomendações

Abordagens possíveis:

- NLP com **TF-IDF**
- **Embeddings semânticos**
- Classificação supervisionada
- Similaridade textual

Bibliotecas possíveis:

- Scikit-learn
- TensorFlow / PyTorch
- Sentence Transformers


#Modelos de IA usados:

- ollama run hf.co/BioMistral/BioMistral-7B-GGUF:Q4_K_M
- ollama run llama3
- ollama run deepseek-ri


# 🔄 Fluxo do Sistema

```text
Usuário
   │
   ▼
Entrada de sintomas / condição médica
   │
   ▼
Processamento NLP
   │
   ▼
Modelo de IA
   │
   ▼
Busca no dataset de medicamentos
   │
   ▼
Sugestão de possíveis medicamentos

```




   
