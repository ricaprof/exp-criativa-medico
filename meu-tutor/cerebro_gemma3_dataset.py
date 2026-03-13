import os
import time
import re
import pandas as pd
from langchain_ollama import OllamaLLM

# ==========================================
# ⚙️ PAINEL DE CONFIGURAÇÕES DO AGENTE ⚙️
# ==========================================

# 1. PROCESSAMENTO (GPU vs CPU)
USAR_GPU = True 

# 2. "RAM" (MEMÓRIA DE CONTEXTO)
# 2048 (Baixo consumo), 4096 (Médio), 8192 (Alto consumo - atenção com a RTX 3050)
MEMORIA_CONTEXTO = 8192 

# 3. NÚMERO MÁXIMO DE COMPARAÇÕES
MAX_COMPARACOES = 2 

# --- Caminhos dos Arquivos ---
NOME_MODELO_OLLAMA = "analisador-med" 
ARQUIVO_PERGUNTAS = "pergunta1.txt"
ARQUIVO_RESPOSTAS = "resposta1.txt"
CAMINHO_DATASET = r"C:\Users\cryst\Downloads\drug_interactions_dataset\drug_interactions_dataset.csv"

# ==========================================
# 📚 DICIONÁRIO DE SINÔNIMOS (FALTAVA ISSO!)
# ==========================================
SINONIMOS_MEDICOS = {
    "paracetamol": "acetaminophen",
    "vitamina c": "ascorbic acid",
    "dipirona": "dipyrone",
    "aas": "aspirin",
    "ácido acetilsalicílico": "aspirin"
}
# ==========================================

def inicializar_ia():
    camadas_gpu = -1 if USAR_GPU else 0
    return OllamaLLM(
        model=NOME_MODELO_OLLAMA,
        temperature=0,
        num_ctx=MEMORIA_CONTEXTO,
        num_gpu=camadas_gpu,
        stop=["<end_of_turn>"]
    )

def carregar_dataset():
    if not os.path.exists(CAMINHO_DATASET):
        print(f"--- [ERRO] Dataset não encontrado ---")
        return None
    try:
        df = pd.read_csv(CAMINHO_DATASET, sep=None, engine='python', on_bad_lines='warn')
        print("Otimizando dataset para buscas ultra-rápidas...")
        df['coluna_busca'] = df.astype(str).agg(' '.join, axis=1).str.lower()
        return df
    except Exception as e:
        print(f"--- [ERRO CRÍTICO] Falha ao ler o CSV: {e} ---")
        return None

def buscar_no_dataset(par, df):
    if df is None or df.empty: return ""
    
    par_limpo = par.replace('?', '')
    medicamentos_brutos = [m.strip().lower() for m in par_limpo.split('+')]
    
    # Agora o dicionário existe e não vai mais dar erro aqui
    medicamentos = [SINONIMOS_MEDICOS.get(med, med) for med in medicamentos_brutos]
    
    med1 = medicamentos[0]
    med2 = medicamentos[1] if len(medicamentos) > 1 else ""

    # BUSCA ESTRITA: Exige que os dois medicamentos estejam na mesma linha
    if med2:
        mask = df['coluna_busca'].str.contains(med1, na=False) & df['coluna_busca'].str.contains(med2, na=False)
        resultado = df[mask]
    else:
        resultado = pd.DataFrame()
    
    if 'coluna_busca' in resultado.columns:
        resultado = resultado.drop(columns=['coluna_busca'])

    return resultado.head(MAX_COMPARACOES).to_string(index=False)

if __name__ == "__main__":
    print("="*50)
    print("INICIANDO CÉREBRO MÉDICO...")
    print(f"-> Processamento via: {'Placa de Vídeo (GPU)' if USAR_GPU else 'Processador (CPU)'}")
    print(f"-> Memória de Contexto: {MEMORIA_CONTEXTO}")
    print(f"-> Máximo de Comparações no CSV: {MAX_COMPARACOES}")
    print("="*50)

    llm = inicializar_ia()
    df_interacoes = carregar_dataset()
    
    if not os.path.exists(ARQUIVO_PERGUNTAS):
        print(f"\n[ERRO] Crie o arquivo '{ARQUIVO_PERGUNTAS}'.")
    else:
        with open(ARQUIVO_PERGUNTAS, "r", encoding="utf-8") as f:
            lista_pares = [linha.strip() for linha in f.readlines() if linha.strip()]

        resultados = []
        for i, par in enumerate(lista_pares, 1):
            print(f"[{i}/{len(lista_pares)}] Analisando: {par}...")
            
            conhecimento_extra = buscar_no_dataset(par, df_interacoes)
            
            # PROMPT BLINDADO CONTRA ALUCINAÇÕES E IOT
            prompt = f"""
            Você é um assistente farmacêutico rigoroso e direto ao ponto. Seu objetivo é analisar a interação entre {par}.
            
            REGRA 1: Leia os dados do DATASET LOCAL. Se houver conflito documentado entre os DOIS medicamentos, resuma o efeito adverso em português de forma técnica.
            REGRA 2: NÃO use analogias, metáforas, comparações com IoT, sensores ou tecnologia. Seja estritamente médico.
            REGRA 3: Se o DATASET LOCAL estiver VAZIO, responda EXATAMENTE:
            "**[NENHUM CONFLITO DETECTADO]** | Não foram encontradas interações negativas graves nesta base de dados para {par}. O uso conjunto geralmente não apresenta contraindicações documentadas aqui, mas requer avaliação clínica individual."

            DATASET LOCAL (Base de Conhecimento):
            {conhecimento_extra}

            RESPOSTA TÉCNICA EM PORTUGUÊS:
            """
            
            try:
                resposta_ia = llm.invoke(prompt)
                bloco = f"PAR: {par}\n{'-'*20}\n{resposta_ia.strip()}\n{'-'*60}\n"
                resultados.append(bloco)
            except Exception as e:
                print(f"Erro ao analisar o par: {e}")

        with open(ARQUIVO_RESPOSTAS, "w", encoding="utf-8") as f_out:
            f_out.writelines(resultados)
        
        print(f"\n[SUCESSO] Relatório gerado no arquivo: {ARQUIVO_RESPOSTAS}")