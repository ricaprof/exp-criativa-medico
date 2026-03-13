import os
import time
import duckdb
from langchain_ollama import OllamaLLM

# ==========================================
# ⚙️ PAINEL DE CONFIGURAÇÕES DO AGENTE ⚙️
# ==========================================

# 1. PROCESSAMENTO (GPU vs CPU)
USAR_GPU = True 

# 2. "RAM" (MEMÓRIA DE CONTEXTO)
# 4096 é o ponto ideal de equilíbrio para manter a RTX 3050 rápida e estável.
MEMORIA_CONTEXTO = 4096 

# 3. NÚMERO MÁXIMO DE COMPARAÇÕES
MAX_COMPARACOES = 2 

# --- Caminhos dos Arquivos ---
NOME_MODELO_OLLAMA = "analisador-med" 
ARQUIVO_PERGUNTAS = "pergunta1.txt"
ARQUIVO_RESPOSTAS = "resposta1.txt"
CAMINHO_DATASET = r"C:\Users\cryst\Downloads\drug_interactions_dataset\drug_interactions_dataset.csv"

# ==========================================
# 📚 DICIONÁRIO DE SINÔNIMOS 
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

def buscar_com_duckdb(par, caminho_csv):
    if not os.path.exists(caminho_csv):
        return ""

    # Limpa caracteres que quebram o SQL
    par_limpo = par.replace('?', '').replace("'", "")
    medicamentos_brutos = [m.strip().lower() for m in par_limpo.split('+')]
    
    medicamentos = [SINONIMOS_MEDICOS.get(med, med) for med in medicamentos_brutos]
    med1 = medicamentos[0]
    med2 = medicamentos[1] if len(medicamentos) > 1 else ""

    if not med2:
        return ""

    # OTIMIZAÇÃO: LIKE '%...%' permite achar a palavra mesmo que tenha espaços ou sufixos no CSV
    query = f"""
        SELECT interaction_description 
        FROM read_csv_auto('{caminho_csv}', ignore_errors=true)
        WHERE (LOWER(drug_name) LIKE '%{med1}%' AND LOWER(interacting_drug_name) LIKE '%{med2}%')
           OR (LOWER(drug_name) LIKE '%{med2}%' AND LOWER(interacting_drug_name) LIKE '%{med1}%')
        LIMIT {MAX_COMPARACOES}
    """
    
    try:
        resultado_df = duckdb.query(query).df()
        if resultado_df.empty:
            return "" 
        return "\n".join(resultado_df['interaction_description'].tolist())
    except Exception as e:
        print(f"Erro SQL: {e}")
        return ""

if __name__ == "__main__":
    print("="*50)
    print("INICIANDO CÉREBRO MÉDICO (MOTOR DUCKDB)...")
    print(f"-> Processamento via: {'Placa de Vídeo (GPU)' if USAR_GPU else 'Processador (CPU)'}")
    print(f"-> Memória de Contexto: {MEMORIA_CONTEXTO}")
    print(f"-> Máximo de Comparações: {MAX_COMPARACOES}")
    print("="*50)

    llm = inicializar_ia()
    
    if not os.path.exists(ARQUIVO_PERGUNTAS):
        print(f"\n[ERRO] Crie o arquivo '{ARQUIVO_PERGUNTAS}' na pasta do projeto.")
    else:
        with open(ARQUIVO_PERGUNTAS, "r", encoding="utf-8") as f:
            lista_pares = [linha.strip() for linha in f.readlines() if linha.strip()]

        resultados = []
        # O script agora vai iniciar imediatamente, sem tempo de carregamento do arquivo
        for i, par in enumerate(lista_pares, 1):
            print(f"[{i}/{len(lista_pares)}] Analisando: {par}...")
            
            conhecimento_extra = buscar_com_duckdb(par, CAMINHO_DATASET)
            
            # 🔥 O GUARDRAIL: Se o banco não achar nada, o Python responde e a IA é ignorada!
            if not conhecimento_extra:
                resposta_segura = f"**[NENHUM CONFLITO DETECTADO]** | Não foram encontradas interações negativas graves nesta base de dados para {par}. O uso conjunto geralmente não apresenta contraindicações documentadas aqui, mas requer avaliação clínica individual."
                bloco = f"PAR: {par}\n{'-'*20}\n{resposta_segura}\n{'-'*60}\n"
                resultados.append(bloco)
                continue # Pula para a próxima pergunta instantaneamente
            
            # Se achou conflito, aí sim chamamos a IA para traduzir o texto
            prompt = f"""
            Você é um assistente farmacêutico rigoroso.
            
            REGRA 1: Leia os dados do DATASET LOCAL abaixo. Resuma o conflito documentado em português de forma técnica e direta.
            REGRA 2: NÃO use analogias, metáforas ou comparações tecnológicas.
            REGRA 3: NÃO inicie sua resposta com tags como [CONFLITO DETECTADO] ou [DADOS INSUFICIENTES]. Vá direto para a explicação médica.

            DATASET LOCAL (Base de Conhecimento):
            {conhecimento_extra}

            RESPOSTA TÉCNICA EM PORTUGUÊS:
            """
            
            try:
                resposta_ia = llm.invoke(prompt)
                bloco = f"PAR: {par}\n{'-'*20}\n**[CONFLITO DETECTADO]** | {resposta_ia.strip()}\n{'-'*60}\n"
                resultados.append(bloco)
            except Exception as e:
                print(f"Erro ao analisar o par: {e}")

        with open(ARQUIVO_RESPOSTAS, "w", encoding="utf-8") as f_out:
            f_out.writelines(resultados)
        
        print(f"\n[SUCESSO] Relatório gerado no arquivo: {ARQUIVO_RESPOSTAS}")