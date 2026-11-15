# RecipeMatch

Projeto acadêmico de recomendação de receitas desenvolvido em Python e Streamlit.

## Sobre o projeto

O RecipeMatch foi desenvolvido para a disciplina Oficina de Desenvolvimento de
Sistemas I. O objetivo é recomendar receitas com base nas preferências dos usuários.

Para isso, o sistema analisa receitas que cada pessoa já avaliou e procura outras
com características semelhantes, como ingredientes, categorias e descrições.
Também é possível usar filtros e consultar explicações sobre as recomendações.

## Funcionalidades

- Seleção de um usuário com histórico suficiente para a demonstração;
- Consulta das interações e das notas válidas no histórico;
- Recomendações personalizadas e explicações relacionadas ao conteúdo das receitas;
- Filtros para ingredientes a evitar, ingredientes disponíveis e tempo de preparo;
- Escolha da quantidade de recomendações;
- Opção MMR para equilibrar relevância e diversidade.

## Como o sistema funciona

1. Carrega as receitas e as avaliações.
2. Identifica as preferências a partir das avaliações do usuário.
3. Compara essas preferências com as informações de outras receitas.
4. Remove receitas que o usuário já conhece.
5. Aplica os filtros escolhidos.
6. Mostra as recomendações e suas explicações.

O TF-IDF ajuda a identificar palavras importantes nas receitas. A similaridade do
cosseno compara as informações das receitas com as preferências do usuário. A opção
MMR pode reorganizar os resultados para equilibrar compatibilidade e diversidade.

Na preparação dos textos, o projeto normaliza acentos, remove palavras comuns em
português e inglês e aplica stemming Snowball em inglês. Esses passos ajustam os
termos usados pelo TF-IDF; não alteram a construção do perfil nem os scores.

## Dados utilizados

O projeto usa o conjunto
[Food.com Recipes and Reviews, disponível no Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews).
São necessários:

- `data/raw/recipes.parquet`: catálogo e informações textuais das receitas;
- `data/raw/reviews.parquet`: interações entre usuários e receitas, incluindo notas
  e datas.

Notas de 1 a 5 são avaliações por estrelas. O valor 0 representa uma interação sem
nota, não uma avaliação negativa. Essas interações continuam no histórico para que
receitas já vistas não sejam recomendadas novamente, mas não entram na média nem no
perfil como avaliação por estrelas.

Os arquivos de dados não são distribuídos pelo repositório. A contagem registrada
no resumo local do conjunto é de 522.517 receitas e 1.401.982 interações. Os números
podem variar conforme a versão baixada.

## Como instalar e executar

Use o Windows PowerShell e execute os comandos a partir da pasta em que deseja
salvar o projeto. É necessário ter Git e Python 3.10 ou mais recente.

1. Baixe o repositório:

   ```powershell
   git clone https://github.com/Nevrand/filtragem-baseada-conteudo.git
   ```

2. Entre na pasta do projeto:

   ```powershell
   cd filtragem-baseada-conteudo
   ```

3. Crie um ambiente virtual:

   ```powershell
   python -m venv .venv
   ```

4. Ative o ambiente virtual:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   No Prompt de Comando, use `.\.venv\Scripts\activate.bat`.

5. Instale as dependências:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Baixe o conjunto [Food.com Recipes and Reviews no
   Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews).

7. Coloque `recipes.parquet` e `reviews.parquet` na pasta `data/raw/`:

   ```powershell
   New-Item -ItemType Directory -Force data\raw
   ```

   O Kaggle disponibiliza os arquivos Parquet diretamente; não é preciso convertê-los.
   Os dados são locais e não são incluídos no repositório.

8. Inicie o Streamlit:

   ```powershell
   python -m streamlit run app/main.py
   ```

   O navegador normalmente abre em `http://localhost:8501`.

## Como utilizar

1. Escolha um usuário na lista.
2. Consulte o histórico de interações e, se desejar, abra o histórico completo.
3. Na barra lateral, configure ingredientes a evitar, ingredientes disponíveis e
   tempo máximo.
4. Defina quantas receitas deseja receber.
5. Opcionalmente, ative MMR e ajuste o equilíbrio entre relevância e diversidade.
6. Selecione **Gerar recomendações**.
7. Consulte as receitas, a posição no ranking e as explicações apresentadas.

## Notebook exploratório

O notebook [notebooks/exploratory_analysis.ipynb](notebooks/exploratory_analysis.ipynb)
resume o tamanho e a qualidade dos dados e a distribuição das notas, além de
gerar gráficos de interações por usuário, categorias, ingredientes e tempo de preparo.
As amostras usadas nos gráficos são reprodutíveis e não alteram os dados originais.

## Avaliação dos resultados

O RecipeMatch foi comparado com dois métodos de referência: recomendações aleatórias
e receitas mais populares no conjunto de treino. Para a avaliação, foram consideradas
relevantes as receitas do teste com nota 4 ou 5 que estão no catálogo e não aparecem
no treino. A avaliação usa a semente 42 e uma divisão cronológica quando há datas
válidas; caso contrário, usa uma divisão aleatória reprodutível.

- **Precision:** quantas recomendações apresentadas eram relevantes.
- **Recall:** quantas receitas relevantes foram encontradas.
- **nDCG:** se as receitas relevantes apareceram em boas posições.
- **Coverage:** quanto do catálogo apareceu nas recomendações.

A tabela resume duas execuções com 100 usuários, semente 42 e Top-10. Em cada uma,
83 usuários entraram nas médias; 17 foram excluídos por não terem itens relevantes
elegíveis no teste ou um perfil informativo. Os resultados dependem dos usuários
escolhidos e da separação entre treino e teste.

| Método | P@5 | R@5 | nDCG@5 | P@10 | R@10 | nDCG@10 | Coverage histórico | Coverage com stemming |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RecipeMatch-TF-IDF | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,001525 | 0,001539 |
| Aleatório | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,001588 | 0,001588 |
| Popularidade | 0,004819 | 0,012195 | 0,011688 | 0,006024 | 0,032275 | 0,017641 | 0,000031 | 0,000031 |

Na amostra registrada, as métricas de recuperação do modelo de conteúdo ficaram em
zero, enquanto o método por popularidade encontrou alguns itens relevantes. Esse
resultado é mantido como foi medido; ele não representa todos os usuários do conjunto.

## Como testar o projeto

Há três maneiras de verificar o projeto:

### A. Testes automatizados

Na raiz do repositório, execute:

```powershell
python -m pytest -q
```

Os testes cobrem carregamento, pré-processamento, perfil do usuário, similaridade,
recomendações, filtros, MMR, métricas, baselines e avaliação experimental. Eles usam
dados pequenos e arquivos Parquet temporários, não precisam dos arquivos do Kaggle e
não executam a avaliação completa com todos os usuários.

### B. Testes rápidos

Para executar os testes rápidos, sem os casos identificados por esses termos nos nomes:

```powershell
python -m pytest -q -k "not parquet and not dataset_report and not experiment_runner and not qualitative_analysis"
```

Na revisão atual, esse filtro seleciona 38 dos 44 testes. Ele é uma verificação
rápida, mas não substitui a suíte completa.

### C. Teste manual da interface

1. Inicie o Streamlit com o comando da seção de instalação.
2. Selecione um usuário e consulte o histórico.
3. Configure os filtros e gere recomendações.
4. Ative MMR e confira a nova ordem, as posições e as explicações.

## Como reproduzir os resultados

Os comandos abaixo devem ser executados na raiz do projeto e precisam dos dois
arquivos Parquet em `data/raw/`. Cada comando grava arquivos na pasta indicada. Use
uma pasta nova ou escolha outro nome para não sobrescrever resultados existentes.

### A. Avaliação pequena

```powershell
python -m src.evaluation.run_experiments --max-users 3 --random-state 42 --top-n 10 --output-dir results/teste-rapido
```

Esse comando limita a avaliação a três usuários para verificar o fluxo. Ainda precisa
preparar o catálogo completo e construir o modelo, então o tempo não foi medido e pode
ser maior que o dos testes rápidos. As métricas dessa amostra não são resultados
oficiais do trabalho.

### B. Reprodução da amostra do relatório

```powershell
python -m src.evaluation.run_experiments --max-users 100 --random-state 42 --top-n 10 --output-dir results/reproducao-100
```

Essa execução avalia 100 usuários e pode demorar significativamente mais por precisar
preparar o catálogo e gerar recomendações. O tempo não foi medido nesta revisão. As
métricas e a configuração são salvas em `results/reproducao-100/`.

### C. Relatório dos dados

```powershell
python -m src.analysis.dataset_report --output-dir results/analise-dados
```

Gera um resumo estatístico do catálogo e das avaliações em `results/analise-dados/`.

### D. Exemplos qualitativos

```powershell
python -m src.evaluation.qualitative_analysis --output results/analise-qualitativa.csv
```

Gera um CSV com históricos, recomendações e explicações de até três usuários. Esse
comando também prepara o catálogo e o modelo; pode demorar e seu tempo não foi medido
nesta revisão.

## Estrutura do projeto

```text
app/                  Interface Streamlit
data/raw/             Arquivos Parquet obtidos localmente
notebooks/            Notebook de análise exploratória
results/              Relatórios e resultados de avaliação
src/analysis/         Resumo descritivo dos dados
src/data/             Leitura e preparação dos dados
src/evaluation/       Divisão, métricas, baselines e experimentos
src/features/         TF-IDF e perfil individual
src/recommendation/   Similaridade, filtros, ranking e MMR
tests/                Testes automatizados
```

## Tecnologias utilizadas

- **Python:** implementação do projeto;
- **Streamlit:** interface interativa;
- **Pandas e NumPy:** manipulação dos dados e cálculos;
- **scikit-learn:** representação textual TF-IDF;
- **SciPy:** operações com matrizes esparsas;
- **PyArrow:** leitura dos arquivos Parquet;
- **NLTK:** stemming Snowball em inglês;
- **Matplotlib e IPython:** gráficos e execução do notebook exploratório;
- **pytest e pytest-cov:** testes e medição de cobertura.

## Limitações

- A recomendação depende da qualidade e da quantidade de informações disponíveis
  nas receitas.
- A demonstração lista usuários com pelo menos 50 interações, e um perfil
  personalizado precisa conter avaliações informativas.
- Receitas com textos parecidos podem aparecer próximas no ranking; MMR oferece uma
  alternativa de ordenação, mas não garante diversidade perfeita.
- Os resultados de avaliação dependem da divisão treino/teste e da amostra usada.
  Métricas offline não substituem a avaliação das recomendações por pessoas.

## Integrantes

- **Alexander** — Dados
- **Geovanna** — Relatório e apresentação
- **Thercio** — Frontend
