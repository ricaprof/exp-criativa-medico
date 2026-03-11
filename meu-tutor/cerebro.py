import os
import time
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. DEFINIÇÃO DOS MODELOS ---
# Nome do modelo criado via Modelfile para o Gemma 3
NOME_MODELO_OLLAMA = "meu-tutor-gemma" 
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

# --- 2. CONFIGURAÇÃO DE CAMINHOS ---
CAMINHO_DOCUMENTOS_LOCAL = r"C:\Users\cryst\.ollama\meu-tutor\meu-tutor\documentos"
DB_LOCAL_STORAGE = "vectorstore/db_faiss"
ARQUIVO_PERGUNTAS = "perguntas.txt"
ARQUIVO_RESPOSTAS = "respostas.txt"

def criar_base_conhecimento():
    """Lê os ficheiros locais e transforma em base de dados para a IA"""
    print(f"Processando documentos de: {CAMINHO_DOCUMENTOS_LOCAL}")
    loader = DirectoryLoader(CAMINHO_DOCUMENTOS_LOCAL, glob="**/*.txt", loader_cls=TextLoader)
    documentos = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos = text_splitter.split_documents(documentos)

    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    vectorstore = FAISS.from_documents(textos, embeddings)
    vectorstore.save_local(DB_LOCAL_STORAGE)
    return vectorstore

def inicializar_ia():
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    if os.path.exists(DB_LOCAL_STORAGE):
        vectorstore = FAISS.load_local(DB_LOCAL_STORAGE, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = criar_base_conhecimento()

    # Configuração otimizada para Gemma 3 4B na RTX 3050
    llm = OllamaLLM(
        model=NOME_MODELO_OLLAMA,
        temperature=0,
        num_ctx=2048,
        stop=["<end_of_turn>", "SINTOMAS:"]
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2}), llm

def limpar_resposta(texto):
    """Remove emojis e caracteres especiais não técnicos da resposta"""
    # Remove emojis e símbolos especiais (Unicode range de emojis)
    texto_limpo = re.sub(r'[^\x00-\x7f]', r'', texto)
    return texto_limpo.strip()
if __name__ == "__main__":
    retriever, llm = inicializar_ia()
    
    # IMPORTANTE: Mude o nome para o novo modelo criado
    NOME_MODELO_OLLAMA = "analisador-med"

    if not os.path.exists(ARQUIVO_PERGUNTAS):
        print(f"\n[ERRO] Crie o arquivo '{ARQUIVO_PERGUNTAS}' com os pares (ex: Aspirina + Varfarina).")
    else:
        with open(ARQUIVO_PERGUNTAS, "r", encoding="utf-8") as f:
            lista_pares = [linha.strip() for linha in f.readlines() if linha.strip()]

        print(f"\nIniciando análise de {len(lista_pares)} interações...")
        resultados = []

        # REMOVIDO O LOOP DUPLICADO
        for i, par in enumerate(lista_pares, 1):
            print(f"[{i}/{len(lista_pares)}] Analisando: {par}...")
            
            try:
                # O retriever PRECISA buscar um novo contexto para cada par!
                docs = retriever.invoke(f"Interação entre {par}")
                contexto = "\n".join([d.page_content for d in docs])
                
                # Prompt limpo para o analisador-med
                prompt = f"Verificar compatibilidade: {par}\nCONTEXTO: {contexto}"
                
                # CHAMA A IA UMA VEZ
                resposta_ia = llm.invoke(prompt)
                
                # Limpa emojis e símbolos não-técnicos
                resposta_final = limpar_resposta(resposta_ia)
                
                bloco = f"PAR ANALISADO: {par}\nAVALIAÇÃO: {resposta_final}\n{'-'*60}\n"
                resultados.append(bloco)
                
                time.sleep(0.5) 

            except Exception as e:
                print(f"Erro no par {i}: {e}")

        # Salva o relatório final após sair do loop
        with open(ARQUIVO_RESPOSTAS, "w", encoding="utf-8") as f_out:
            f_out.writelines(resultados)
        
        print(f"\n[SUCESSO] Análise concluída! Verifique o arquivo: {ARQUIVO_RESPOSTAS}")