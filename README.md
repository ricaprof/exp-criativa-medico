Medical Prescription AI: Analisador de Compatibilidade
Este projeto consiste em um sistema de Inteligência Artificial para análise e comparação farmacológica. Utilizando técnicas de RAG (Retrieval-Augmented Generation), o sistema consulta uma base de dados real de medicamentos para verificar interações medicamentosas e explicar por que dois fármacos podem ou não ser administrados juntos.

 Aviso Importante: Este projeto possui finalidade estritamente educacional e de pesquisa. Os resultados gerados são simulações baseadas em dados e não devem ser utilizados para diagnóstico ou prescrição médica real. Sempre consulte um profissional de saúde.

Objetivo
Desenvolver um motor de busca inteligente capaz de:

Analisar Pareamentos: Verificar se dois medicamentos são compatíveis ou se geram conflito.

Exposição Técnica: Explicar o mecanismo bioquímico da interação (ex: inibição enzimática, receptores).

Tradução Conceitual: Gerar analogias com o mundo de IoT e Engenharia de Hardware para facilitar o entendimento técnico dos mecanismos.

Busca Semântica: Utilizar uma base de conhecimento local para evitar "alucinações" da IA.

Base de Dados (Dataset)
O projeto utiliza o DRUG INTERACTIONS DATA, que provê uma base sólida com:

~1.000.000 de interações medicas.

- **Name & Composition:** Identificação de princípios ativos para cruzamento de dados.
- **Therapeutic & Chemical Class:** Classificação para identificação de redundância terapêutica (ex: tomar dois AINES juntos).
- **Side Effects:** Base de dados para identificar reações adversas potencializadas por interações.
- **Usage Instructions:** Contexto sobre administração para evitar conflitos de absorção.

Fonte: DruhBank

🏗️ Arquitetura do Sistema
O sistema é estruturado em três camadas principais:

1️⃣ Processamento e Vetorização (RAG)
Document Loading: Leitura de arquivos .txt contendo o conhecimento farmacológico.

Vetorização: Uso do modelo all-MiniLM-L6-v2 para transformar textos em vetores matemáticos.

Vector Store: Armazenamento e busca eficiente de similaridade utilizando FAISS.

2️⃣ Motor de Inferência (Local LLM)
Utilizamos o Ollama para rodar modelos de linguagem de ponta localmente, garantindo privacidade e performance:

Gemma 3 (4B): Modelo principal para extração técnica e lógica.

Llama 3: Alternativa para processamento de linguagem natural.

DeepSeek-R1: Utilizado para raciocínio lógico complexo em interações.

3️⃣ Filtro de Saída e Higienização
Limpeza ASCII: Script em Python que remove emojis e caracteres especiais para manter o relatório estritamente técnico e legível em qualquer terminal.

🔄 Fluxo de Funcionamento
Plaintext
Entrada (Remédio A + Remédio B)
       │
       ▼
Busca Semântica (Base de Dados FAISS)
       │
       ▼
Recuperação de Contexto Técnico
       │
       ▼
Prompt Engineering (Modelfile Customizado)
       │
       ▼
Inferência no Gemma 3 (Ollama)
       │
       ▼
Filtro de Limpeza (Python Regex)
       │
       ▼
Saída: **[STATUS]** | Motivo Técnico | Analogia IoT
 Tecnologias Utilizadas
Linguagem: Python 3.10+

IA/LLM: Ollama (Gemma 3, Llama 3, DeepSeek)

Framework RAG: LangChain

Banco Vetorial: FAISS

Processamento de Dados: Pandas, Re (Regex)

Hardware Otimizado: Suporte para aceleração via GPU (NVIDIA RTX 3050 via CUDA)

📝 Exemplo de Saída Técnica
PAR ANALISADO: Varfarina + Aspirina

AVALIAÇÃO: [CONFLITO DETECTADO] | Motivo técnico: A aspirina inibe a agregação plaquetária enquanto a varfarina antagoniza a vitamina K. A administração conjunta potencializa o risco de hemorragias severas. | Analogia IoT: É como um conflito de interrupção (IRQ) onde dois dispositivos tentam acessar o barramento de controle simultaneamente, causando corrupção de dados e falha crítica no sistema.

Como Rodar o Projeto
Certifique-se de ter o Ollama instalado.

Crie o modelo customizado: ollama create analisador-med -f Modelfile.

Adicione suas notas na pasta /documentos.

Execute o motor principal: python cerebro.py.

Desenvolvido por: Crystofer Samuel, Murilo Chandelier, Ricardo Viena e Ricardo Ryu

Área: Inteligência Artificial Aplicada à Saúde e Engenharia.