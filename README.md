# RecipeMatch

Projeto acadêmico de recomendação de receitas desenvolvido em Python e Streamlit.

## Sobre o projeto

O RecipeMatch foi desenvolvido para a disciplina Oficina de Desenvolvimento de
Sistemas I. O projeto recomenda receitas de acordo com o histórico individual de
cada usuário: receitas que ele avaliou ajudam a identificar características de que
gosta ou não gosta.

O sistema compara essas preferências com as informações do catálogo, como nomes,
descrições, categorias, palavras-chave e ingredientes. A interface também permite
filtrar resultados e consultar por que uma receita foi recomendada. A personalização
é baseada no conteúdo das receitas, não nas avaliações de outros usuários.

## Funcionalidades

- Seleção de um usuário com histórico suficiente para a demonstração;
- Consulta das interações e das notas válidas no histórico;
- Recomendações personalizadas e explicações relacionadas ao conteúdo das receitas;
- Filtros para ingredientes a evitar, ingredientes disponíveis e tempo de preparo;
- Escolha da quantidade de recomendações;
- Opção MMR para equilibrar relevância e diversidade.

## Como o sistema funciona

1. Carrega as receitas e as interações registradas no conjunto de dados.
2. Prepara os textos das receitas e combina informações como ingredientes e
   categorias.
3. Usa as notas válidas do histórico para identificar preferências individuais.
4. Compara o perfil do usuário com as receitas que ele ainda não viu.
5. Aplica os filtros selecionados e ordena as candidatas por compatibilidade.
6. Exibe as recomendações e os termos que contribuíram para cada resultado.

O TF-IDF ajuda a destacar palavras importantes nas informações das receitas. A
similaridade do cosseno compara os vetores de texto do perfil e das receitas. Notas
altas e baixas influenciam o perfil em sentidos diferentes; uma nota neutra ou uma
interação sem nota não acrescenta preferência por estrelas. A opção MMR pode
reordenar o resultado para equilibrar compatibilidade e diversidade.

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

Execute os comandos a partir da pasta raiz do repositório. É necessário ter Git e
Python 3.10 ou mais recente.

1. Obtenha uma cópia do projeto:

   ```powershell
   git clone https://github.com/Nevrand/filtragem-baseada-conteudo.git
   cd filtragem-baseada-conteudo
   ```

2. Crie um ambiente virtual:

   ```powershell
   python -m venv .venv
   ```

3. Ative o ambiente no Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   No Prompt de Comando do Windows, use `.\.venv\Scripts\activate.bat`.

4. Instale as dependências:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Baixe o dataset na página do
   [Food.com Recipes and Reviews no Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews).
   Depois de extrair o conteúdo, coloque os dois arquivos Parquet na pasta
   `data/raw/` e confira os nomes:

   ```text
   data/raw/recipes.parquet
   data/raw/reviews.parquet
   ```

   É necessário ter acesso ao Kaggle para baixar o conjunto. Os Parquet são arquivos
   locais e não são versionados pelo Git.

6. Inicie a interface:

   ```powershell
   python -m streamlit run app/main.py
   ```

7. Abra no navegador o endereço informado pelo Streamlit, normalmente
   `http://localhost:8501`.

## Como utilizar

1. Escolha um usuário na lista.
2. Consulte o histórico de interações e, se desejar, abra o histórico completo.
3. Na barra lateral, configure ingredientes a evitar, ingredientes disponíveis e
   tempo máximo.
4. Defina quantas receitas deseja receber.
5. Opcionalmente, ative MMR e ajuste o equilíbrio entre relevância e diversidade.
6. Selecione **Gerar recomendações**.
7. Consulte as receitas, a posição no ranking e as explicações apresentadas.

## Avaliação dos resultados

Os experimentos comparam o RecipeMatch com dois métodos de referência: seleção
aleatória com semente fixa e ordenação por popularidade no conjunto de treino. A
divisão por usuário separa pares usuário-receita entre treino e teste. O perfil e a
popularidade usam somente o treino; interações de teste não são usadas para gerar as
recomendações. São relevantes as receitas do teste com nota válida 4 ou 5 que
pertençam ao catálogo e não tenham aparecido no treino. O experimento usa semente 42
por padrão e prefere uma divisão cronológica quando os pares usuário-receita têm
datas válidas; nos demais casos, a divisão aleatória é reprodutível.

- **Precision@5 e Precision@10** indicam a proporção de itens relevantes nas
  primeiras cinco ou dez posições.
- **Recall@5 e Recall@10** indicam quantos dos itens relevantes do teste foram
  encontrados nessas posições.
- **nDCG@5 e nDCG@10** também consideram a posição dos itens relevantes.
- **Coverage** informa a fração do catálogo elegível que apareceu nas listas.

Uma avaliação registrada em `results/report/` usou uma amostra de 100 usuários,
semente 42, divisão cronológica e Top-10. Foram avaliados 83 usuários; 17 foram
excluídos das médias por não terem itens relevantes elegíveis ou perfil informativo.
Os resultados abaixo são dessa amostra, não uma estimativa do desempenho de todos os
usuários:

| Método | Precision@5 | Recall@5 | nDCG@5 | Precision@10 | Recall@10 | nDCG@10 | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| RecipeMatch-TF-IDF | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,001525 |
| Aleatório | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,000000 | 0,001588 |
| Popularidade | 0,004819 | 0,012195 | 0,011688 | 0,006024 | 0,032275 | 0,017641 | 0,000031 |

Esses valores são os resultados já registrados nos arquivos de relatório; não foram
recalculados durante esta atualização. Para executar novamente uma amostra de 100
usuários sem sobrescrever o relatório existente:

```powershell
python -m src.evaluation.run_experiments --max-users 100 --random-state 42 --top-n 10 --output-dir results/reproduction-100-users
```

O programa também aceita uma execução sem `--max-users`, que considera os usuários
disponíveis e pode levar mais tempo. Os arquivos de saída registram a configuração,
os usuários selecionados e as métricas por usuário.

## Análise qualitativa

O módulo de análise gera exemplos de até três usuários, com histórico, recomendações,
explicações e uma indicação de itens relevantes do teste que não apareceram no Top-N.
Na cópia local analisada, `results/sample/qualitative_examples.csv` registra uma
amostra de três usuários e 15 recomendações; nenhuma delas coincidiu com um item
relevante identificado no teste. Esse arquivo local não é versionado e o resultado
descreve somente a amostra salva.

Para gerar novos exemplos sem substituir esse arquivo:

```powershell
python -m src.evaluation.qualitative_analysis --output results/reproduction-qualitative.csv
```

## Estrutura do projeto

```text
app/                  Interface Streamlit
data/raw/             Arquivos Parquet obtidos localmente
notebooks/            Espaço para análises em notebook
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

## Referências

- Food.com Recipes and Reviews — Kaggle:
  <https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews>
- Python: <https://www.python.org/>
- pandas: <https://pandas.pydata.org/>
- NumPy: <https://numpy.org/>
- scikit-learn: <https://scikit-learn.org/>
- SciPy: <https://scipy.org/>
- Streamlit: <https://streamlit.io/>
- PyArrow: <https://arrow.apache.org/docs/python/>
