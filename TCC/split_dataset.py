import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_json("drug_interactions_llm.jsonl", lines=True)

train, test = train_test_split(df, test_size=0.1, random_state=42)
train, val = train_test_split(train, test_size=0.1, random_state=42)

train.to_json("train.jsonl", orient="records", lines=True)
val.to_json("validation.jsonl", orient="records", lines=True)
test.to_json("test.jsonl", orient="records", lines=True)

print("Dataset dividido com sucesso")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))