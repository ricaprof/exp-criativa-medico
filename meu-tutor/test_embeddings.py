from langchain_huggingface import HuggingFaceEmbeddings
import os
MODEL = 'manifests/hf.co/BioMistral/BioMistral-7B-GGUF/Q4_K_M'
print('isdir', os.path.isdir(MODEL))
try:
    emb = HuggingFaceEmbeddings(model_name=MODEL)
    print('created')
except Exception as e:
    print('error', e)
