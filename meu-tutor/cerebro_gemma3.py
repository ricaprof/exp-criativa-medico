import os
import time
import re
from langchain_ollama import OllamaLLM

# --- 1. DEFINIÇÃO DOS MODELOS ---
NOME_MODELO_OLLAMA = "analisador-med" 

# --- 2. CONFIGURAÇÃO DE CAMINHOS ---
ARQUIVO_PERGUNTAS = "perguntas.txt"
ARQUIVO_RESPOSTAS = "respostas.txt"

def inicializar_ia():
    """Inicializa apenas o LLM, sem banco de dados vetorial."""
    # Configuração otimizada para a sua RTX 3050
    llm = OllamaLLM(
        model=NOME_MODELO_OLLAMA,
        temperature=0,
        num_ctx=2048,
        stop=["<end_of_turn>", "SINTOMAS:"]
    )
    return llm

def limpar_resposta(texto):
    """Remove emojis e caracteres especiais não técnicos da resposta"""
    texto_limpo = re.sub(r'[^\x00-\x7f]', r'', texto)
    return texto_limpo.strip()

if __name__ == "__main__":
    llm = inicializar_ia()
    
    if not os.path.exists(ARQUIVO_PERGUNTAS):
        print(f"\n[ERRO] Crie o arquivo '{ARQUIVO_PERGUNTAS}' com os pares (ex: Aspirina + Varfarina).")
    else:
        with open(ARQUIVO_PERGUNTAS, "r", encoding="utf-8") as f:
            lista_pares = [linha.strip() for linha in f.readlines() if linha.strip()]

        print(f"\nIniciando análise direta de {len(lista_pares)} interações...")
        resultados = []

        for i, par in enumerate(lista_pares, 1):
            print(f"[{i}/{len(lista_pares)}] Analisando: {par}...")
            
            try:
                # Agora enviamos apenas os medicamentos, sem injetar contexto de arquivos
                prompt = f"Medicamentos: {par}"
                
                # Chama o modelo
                resposta_ia = llm.invoke(prompt)
                
                # Limpa a resposta
                resposta_final = limpar_resposta(resposta_ia)
                
                bloco = f"PAR ANALISADO: {par}\nAVALIAÇÃO: {resposta_final}\n{'-'*60}\n"
                resultados.append(bloco)
                
                time.sleep(0.5) 

            except Exception as e:
                print(f"Erro no par {i}: {e}")

        # Salva o relatório final
        with open(ARQUIVO_RESPOSTAS, "w", encoding="utf-8") as f_out:
            f_out.writelines(resultados)
        
        print(f"\n[SUCESSO] Análise concluída! Verifique o arquivo: {ARQUIVO_RESPOSTAS}")