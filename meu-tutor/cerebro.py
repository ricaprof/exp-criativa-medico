import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. DEFINIÇÃO DOS MODELOS ---
# Aqui defines qual modelo o Ollama vai usar e qual modelo fará a leitura dos ficheiros.
# Para rodar com a versão quantizada em Q4_K_M, ajustamos ambos a seguir.
# O nome usado em `OllamaLLM` deve corresponder ao modelo instalado (p.ex. "biomistral-7b-gguf-q4_k_m").
NOME_MODELO_OLLAMA = "meu-tutor-bio"  # ex: "llama3" ou "deepseek-r1" também são válidos
# Modelo de embeddings (use um modelo compatível com sentence-transformers,
# por exemplo "all-MiniLM-L6-v2" ou outro da biblioteca HuggingFace).  
# A versão Q4_K_M não é válida para embeddings e apenas serve ao LLM quantizado.
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

# --- 2. CONFIGURAÇÃO DA BASE DE DADOS LOCAL ---
# Altera para o caminho da pasta onde estão os teus documentos no teu PC
CAMINHO_DOCUMENTOS_LOCAL = r"C:\TeuCaminho\Para\Os\Arquivos"
# Caminho onde o índice FAISS será guardado (base de dados processada)
DB_LOCAL_STORAGE = "vectorstore/db_faiss"

def criar_base_conhecimento():
    """Lê os ficheiros locais e transforma em base de dados para a IA"""
    print(f"A processar documentos de: {CAMINHO_DOCUMENTOS_LOCAL}")
    
    # Carregador de diretório (ajusta o glob conforme a extensão dos teus ficheiros)
    loader = DirectoryLoader(CAMINHO_DOCUMENTOS_LOCAL, glob="**/*.txt", loader_cls=TextLoader)
    documentos = loader.load()

    # Divisão do texto em partes menores
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos = text_splitter.split_documents(documentos)

    # Inicializa o modelo de embeddings (o que "entende" o significado das palavras)
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    
    # Cria o banco de dados vetorial e guarda localmente
    vectorstore = FAISS.from_documents(textos, embeddings)
    vectorstore.save_local(DB_LOCAL_STORAGE)
    return vectorstore

def inicializar_ia():
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    
    # Se a base já existir, carrega-a. Se não, cria uma nova do zero.
    if os.path.exists(DB_LOCAL_STORAGE):
        vectorstore = FAISS.load_local(DB_LOCAL_STORAGE, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = criar_base_conhecimento()

    # Define o modelo LLM através do Ollama
    llm = OllamaLLM(model=NOME_MODELO_OLLAMA)
    
    return vectorstore.as_retriever(), llm

if __name__ == "__main__":
    retriever, llm = inicializar_ia()
    
    pergunta = input("O que desejas saber sobre a tua base de dados? ")
    
    # Recupera contexto dos teus ficheiros
    contexto_docs = retriever.invoke(pergunta)
    contexto_texto = "\n".join([doc.page_content for doc in contexto_docs])
    
    # Prompt REFORMULADO para ser agressivo na extração
    prompt = f"""
    [CONTEXTO TÉCNICO]:
    {contexto_texto}

    Sintomas do Usuário: {pergunta}
    Ação: Retorne apenas o nome do fármaco, mecanismo e analogia IoT conforme o Modelfile.
    Resposta:"""
    
    print("\n--- Resposta da IA ---")
    print(llm.invoke(prompt))