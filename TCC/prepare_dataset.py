import pandas as pd
import json
import csv

arquivo = "drug_interactions_dataset.csv"

# detectar separador automaticamente
with open(arquivo, "r", encoding="utf-8") as f:
    sample = f.read(5000)
    dialect = csv.Sniffer().sniff(sample)
    separador = dialect.delimiter

print("Separador detectado:", separador)

# carregar dataset
df = pd.read_csv(arquivo, sep=separador)

print("Colunas encontradas:")
print(df.columns)

output_file = "drug_interactions_llm.jsonl"

with open(output_file, "w", encoding="utf-8") as f:

    for row in df.itertuples():

        prompt = f"What is the interaction between {row.drug_name} and {row.interacting_drug_name}?"
        response = row.interaction_description

        data = {
            "input": prompt,
            "output": response
        }

        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print("Conversão finalizada!")
print("Total de exemplos:", len(df))