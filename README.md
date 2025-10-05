# Kotara Framework

<p align="center">
  </p>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)


> Um framework de fusão adaptativa para combinar dois modelos de linguagem causal (LLMs) e avaliar a perplexidade (PPL) resultante.

**Kotara** é uma ferramenta de pesquisa que explora a questão: "A 'mente' combinada de dois LLMs pode superar o desempenho de cada um individualmente?". Inspirado na fusão Potara, este framework permite combinar as previsões de dois modelos em tempo real usando diversas estratégias, desde uma simples média até métodos adaptativos baseados na confiança de cada modelo.

---

## 📖 Sumário

* [ Sobre o Projeto](#-sobre-o-projeto)
* [ Resultados Preliminares e Heurísticas](#-resultados-preliminares-e-heurísticas)
* [ Recursos](#-recursos)
* [ Instalação](#-instalação)
* [ Como Usar](#-como-usar)
* [ As Estratégias de Fusão](#-as-estratégias-de-fusão)
* [ Roadmap](#️-roadmap)
* [ Licença](#-licença)
* [ Autor](#️-autor)

---

## ✨ Sobre o Projeto

O Kotara nasce da visão de poder fundir múltiplos modelos de linguagem em uma única entidade coesa, capaz de aprimorar o desempenho em tarefas downstream. A abordagem principal é um framework modular que opera através da **fusão de logits em tempo real**, sem a necessidade de modificar ou mesclar os pesos dos modelos, tornando o processo leve e flexível.

Por enquanto, o framework foca na avaliação intrínseca via **Perplexidade (PPL)**, mas a arquitetura é projetada para ser extensível.

É importante notar que esta é uma **versão beta**, e os experimentos iniciais foram conduzidos em condições específicas. Isso significa que os resultados podem não generalizar para todos os cenários ou pares de modelos. Versões futuras buscarão mitigar essas limitações e aprimorar o desempenho da fusão em uma gama mais ampla de tarefas.

##  Resultados Preliminares e Heurísticas

Testes realizados pelo **GTLM Research** com este framework, embora limitados, revelaram alguns insights e heurísticas iniciais que podem guiar futuras experimentações:

* **O Ponto Ideal de Similaridade:** Houve ganhos observáveis na PPL **se, e somente se**, os modelos possuíam um grau de similaridade arquitetural e não provinham de domínios de conhecimento completamente distintos. A fusão parece se beneficiar de "perspectivas" diferentes sobre um conhecimento em comum.

* **Excesso de Similaridade:** Modelos idênticos ou extremamente similares não apresentaram ganhos significativos. A fusão de `A + A` não gera um `A` melhor, pois não há informação nova sendo introduzida no processo.

* **Excesso de Diferença:** A fusão de modelos de domínios muito distintos pode ser **prejudicial** ao desempenho se não houver uma base de conhecimento comum e robusta entre eles. A fusão pode gerar mais "ruído" do que "sinal".

> **Aviso:** Devido a limitações de tempo e recursos, uma exploração exaustiva para definir as melhores heurísticas ainda não foi concluída. Estes são insights preliminares que servem como um ponto de partida para futuras investigações.

##  Recursos

* **Suporte Amplo:** Compatível com qualquer arquitetura `AutoModelForCausalLM` do Hugging Face (incluindo modelos que exigem `trust_remote_code=True`).
* **Flexível:** Avalie os modelos base individualmente e, em seguida, o resultado da fusão.
* **Estratégias Múltiplas:** Inclui quatro estratégias de fusão na versão beta: `average`, `poe`, `entropy` e `gap`.

---

##  Instalação


```bash
# 1. Clone o repositório
git clone [https://github.com/MadrasLe/kotara-framework.git](https://github.com/MadrasLe/kotara-framework.git)
cd kotara

# 2. (Recomendado) Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

 Como Usar
O framework é operado via linha de comando. Você precisará de dois modelos (locais ou do Hugging Face Hub) e um arquivo de texto (.txt) para avaliação.

Exemplo de uso básico (fusão com média simples):

Bash

python kotara.py \
    --model_a "gpt2" \
    --model_b "distilgpt2" \
    --dataset_path "./caminho/para/seu_texto.txt" \
    --strategy "average" \
    --save_metrics "resultados_average.json"

Bash

python kotara.py \
    --model_a "/path/to/your/local_model_A" \
    --model_b "EleutherAI/gpt-neo-125M" \
    --dataset_path "./data/wiki_pt.txt" \
    --strategy "entropy" \
    --temp_a 0.85 \
    --temp_b 1.1 \
    --dtype "bfloat16" \
    --max_length 2048 \
    --stride 1024 \
    --save_metrics "resultados_entropy.json"

Após a execução, um relatório será impresso no console e as métricas detalhadas serão salvas no arquivo JSON especificado.

## 🧠 As Estratégias de Fusão

O Kotara permite escolher entre diferentes lógicas para combinar os modelos. Cada uma tem uma filosofia diferente sobre como extrair o melhor de ambos os "especialistas".

* **`average`**: A abordagem mais direta e um excelente baseline. Os logits (as previsões brutas) dos dois modelos são simplesmente somados e divididos por dois. É como tirar a média da opinião de dois especialistas.

* **`poe` (Product of Experts)**: Uma técnica mais sofisticada onde as probabilidades dos modelos são multiplicadas (ou, de forma equivalente, seus log-proporções são somados). Este método tende a produzir previsões mais "pontiagudas" e confiantes, especialmente quando ambos os modelos concordam fortemente em uma previsão.

* **`entropy`**: Esta é uma estratégia de fusão **adaptativa** que funciona como um "supervisor de confiança". Para cada token a ser previsto, ela mede a **Entropia de Shannon** da distribuição de probabilidade de cada modelo.
    * **O que é Entropia?** Pense nela como uma medida de "incerteza" ou "surpresa". Uma entropia baixa significa que o modelo está muito confiante em sua previsão (ex: 90% de chance para uma palavra). Uma entropia alta significa que o modelo está incerto (ex: as probabilidades estão muito espalhadas entre várias palavras).
    * **Como funciona?** A estratégia `entropy` dá um peso maior ao modelo que apresenta a **menor incerteza** para aquele token específico. O peso é literalmente `1 / incerteza`. Assim, o "especialista" mais seguro para cada palavra ganha mais voz na decisão final.

* **`gap`**: Outra estratégia **adaptativa** que mede a confiança de uma forma diferente e muito intuitiva: a "margem de vitória" da melhor previsão.
    * **O que é o "Gap"?** É a diferença de probabilidade entre a palavra mais provável (top-1) e a segunda mais provável (top-2).
    * **Como funciona?** Um "gap" grande significa que o modelo não tem dúvidas; sua melhor aposta se destaca claramente das outras. Um "gap" pequeno indica que o modelo está dividido entre duas ou mais opções. A estratégia `gap` dá um peso maior ao modelo que tem a **decisão mais clara e inequívoca**, ou seja, o maior "gap". Ele recompensa a decisão, não apenas a confiança.

 Roadmap
Esta é a versão beta do Kotara. O plano para a v1.0 e além inclui:

[ ] Adicionar mais estratégias de fusão.

[ ] Implementar um modo de "batch processing" para avaliação ainda mais rápida.

[ ] Suporte para fusão de mais de dois modelos.

[ ] Criação de um pacote pip para fácil instalação.

[ ] Adicionar exemplos em um notebook Jupyter.


 Licença
Este projeto é distribuído sob a Licença Apache 2.0. Veja o arquivo LICENSE para mais detalhes.

 Autor
Gabriel (MadrasLe)

GitHub: @MadrasLe

Um projeto da GTLM Research.
