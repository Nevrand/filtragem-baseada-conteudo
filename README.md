# RecipeMatch

Trabalho bimestral de Oficina de Desenvolvimento de Sistemas I.

## Sobre o projeto

RecipeMatch recomenda receitas com base no conteúdo das receitas avaliadas por cada
usuário. O sistema representa nomes, descrições, categorias, palavras-chave e
ingredientes com TF-IDF e compara o perfil aprendido do usuário com as receitas do
catálogo por similaridade de cosseno.

Recomendação por conteúdo personaliza a lista usando características das receitas do
histórico individual. Recomendação por popularidade, usada como baseline, ordena os
itens pelo número de avaliações válidas por estrelas (1 a 5) no conjunto de
treinamento e não personaliza por preferências textuais.

## Objetivo

- Utilizar avaliações para identificar características que agradam ou desagradam.
- Construir um perfil textual individual para cada usuário.
- Recomendar receitas ainda não avaliadas por esse usuário.
- Aplicar filtros opcionais de ingredientes e tempo.
- Explicar recomendações com termos que contribuíram para a compatibilidade.
- Avaliar o método contra baselines reprodutíveis.

## Dados utilizados

O projeto espera o dataset [Food.com Recipes and Reviews no Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews):

- `data/raw/recipes.parquet`: catálogo de receitas. O esquema observado inclui
  `RecipeId`, `Name`, `Description`, `RecipeCategory`, `Keywords`,
  `RecipeIngredientParts`, `PrepTime`, `CookTime`, `TotalTime` e `Images`.
- `data/raw/reviews.parquet`: avaliações, com `ReviewId`, `RecipeId`, `AuthorId`,
  `Rating`, `Review` e `DateSubmitted` (além de campos de autor e data de modificação).

Receitas fornecem os textos e atributos usados no TF-IDF. Cada linha de avaliações
representa uma interação usuário-receita; somente ratings de 1 a 5 são avaliações
por estrelas. No Food.com, rating 0 significa interação sem avaliação por estrelas,
não uma nota negativa. Essas interações são mantidas para identificar receitas já
visitadas, mas não entram em perfis, médias, distribuições de notas ou contagens do
baseline de avaliações. Os Parquet não são versionados; obtenha-os no Kaggle e
coloque-os em `data/raw/`.

O comando `dataset_report.py` recalcula esses valores a partir dos arquivos locais.
No relatório local executado em 12/11/2026, foram encontradas 522.517 receitas,
1.401.982 interações, 1.325.734 avaliações válidas e 76.248 interações com rating 0.
Havia 271.907 usuários distintos e 271.678 receitas com interação; 265.816 receitas
tinham pelo menos uma avaliação válida. A média das notas válidas foi 4,66147 e
520.298 receitas tinham ingredientes. `dataset_summary.json` separa essas contagens;
a média e a distribuição incluem somente notas de 1 a 5. Os valores podem variar
conforme a versão dos Parquet.

## Como as recomendações são geradas

1. Os textos são normalizados e consolidados; ingredientes são mantidos como
   características centrais. Stopwords comuns em inglês são removidas.
2. `TfidfVectorizer` gera uma matriz esparsa, limitada a 5.000 termos no fluxo padrão.
3. O perfil é a soma ponderada dos vetores das receitas avaliadas.
4. O produto escalar entre vetores TF-IDF L2-normalizados corresponde à similaridade
   de cosseno.
5. Receitas vistas pelo usuário e receitas excluídas pelos filtros são removidas.
6. As receitas restantes são ordenadas pelo score e limitadas ao Top-N.
7. A explicação usa os termos com contribuição positiva (`peso no perfil × peso no
   vetor da receita`) e, quando há sobreposição, pode indicar a receita similar do
   histórico.

Para o usuário `u`, o perfil é:

`P_u = normalize(sum_i(weight(r_ui) * TFIDF_i))`

Uma soma nula ou um histórico apenas neutro não gera perfil artificial.

## Perfil do usuário

Os pesos das avaliações válidas são:

| Nota | Peso | Interpretação |
|---:|---:|---|
| 5 | +2 | positiva forte |
| 4 | +1 | positiva |
| 3 | 0 | neutra |
| 2 | -1 | negativa |
| 1 | -2 | negativa forte |

Uma interação com rating 0 não tem peso no perfil. Avaliações válidas repetidas do
mesmo usuário para a mesma receita são agregadas pela média antes de aplicar o peso;
ratings inválidos também não entram nessa média. A interação permanece no histórico
para excluir a receita das recomendações. Uma média igual a 3 não contribui ao
perfil. Médias do histórico usam somente ratings válidos de 1 a 5.

## Filtros

- **Ingredientes a evitar:** remove receitas com o ingrediente como expressão de
  palavras completas, inclusive descrições como “fresh tomato” para a busca “tomato”,
  sem confundir fragmentos como “tomato” com “tomatillo”.
- **Ingredientes disponíveis:** mantém receitas que contenham pelo menos um dos
  ingredientes informados. Essa informação é fornecida manualmente pelo usuário;
  avaliações passadas não indicam o que ele tem em casa.
- **Tempo máximo:** usa `TotalTime`. Se não estiver disponível para uma receita,
  soma `PrepTime` e `CookTime` somente quando ambos são conhecidos. Receitas sem
  duração verificável são excluídas quando o filtro de tempo está ativo.

## Explicabilidade

Os termos explicados são ordenados pela contribuição positiva ao produto perfil–
receita, e os termos de ingredientes são destacados quando fazem parte dos
ingredientes daquela receita. Se não há termos com contribuição positiva, o sistema
informa que a compatibilidade é geral e não atribui ingredientes sem evidência.

Exemplos reais podem ser gerados por `python -m
src.evaluation.qualitative_analysis`; não há exemplo experimental fixado neste README
antes de executar o comando no dataset local.

## Avaliação

Antes da divisão, o avaliador consolida interações repetidas do mesmo par
`AuthorId`–`RecipeId`: a nota consolidada é a média das notas válidas de 1 a 5; se
não existir nenhuma nota válida, preserva 0 quando houver interação sem avaliação por
estrelas. A linha com data mais recente representa os demais campos. Pares, e não
eventos individuais, são separados por usuário. Quando cada par possui ao menos uma
data válida, usa-se a interação válida mais recente do par e os pares cronologicamente
mais recentes ficam no teste; caso contrário, usa divisão aleatória reprodutível por
usuário. Assim, um par nunca aparece nos dois conjuntos.

O perfil e o baseline de popularidade usam exclusivamente o treino. Todas as
interações do treino, inclusive rating 0, excluem receitas já vistas. O conjunto
relevante do teste contém somente receitas com nota válida de 4 ou 5, presentes no
catálogo e ainda não vistas no treino. Positivos fora do catálogo e já vistos são
contabilizados separadamente, mas não entram no denominador do Recall. Os três
modelos recebem as mesmas candidatas e o mesmo conjunto relevante. Usuários sem
relevantes elegíveis ou sem perfil informativo aparecem nos resultados com o motivo
de exclusão e não entram nas médias.

Métricas:

- **Precision@K:** relevantes recomendados nos primeiros K, divididos por K.
- **Recall@K:** relevantes recomendados nos primeiros K, divididos pelo total de
  itens relevantes no teste.
- **nDCG@K:** ganho binário descontado pelo logaritmo da posição, normalizado pelo
  ranking ideal.
- **Coverage:** fração de receitas distintas recomendadas dividida pelo total de receitas
  elegíveis, definido como a união das candidatas dos usuários avaliáveis após
  excluir receitas vistas no treino. O CSV apresenta `coverage` como fração entre 0
  e 1, e exporta seu denominador em `total_eligible_items`.

## Experimentos e resultados

O avaliador compara RecipeMatch TF-IDF, recomendação aleatória com semente fixa e
popularidade calculada exclusivamente nas avaliações de treino. Todos recebem os
mesmos usuários e, para cada usuário, as mesmas candidatas após a exclusão do
histórico de treino.

Uma amostra não representa automaticamente o dataset completo. Experimentos novos
devem ser identificados pelo tamanho da amostra. A execução real com 100 usuários,
semente 42, divisão cronológica e Top-10 selecionou 100 IDs; 83 foram avaliados e
17 excluídos (14 sem itens relevantes elegíveis e 3 sem perfil informativo). Os
resultados médios observados foram:

| Modelo | Precision@5 | Recall@5 | nDCG@5 | Precision@10 | Recall@10 | nDCG@10 | Coverage (fração) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RecipeMatch-TF-IDF | 0 | 0 | 0 | 0 | 0 | 0 | 0,001525 |
| Aleatório | 0 | 0 | 0 | 0 | 0 | 0 | 0,001588 |
| Popularidade | 0,004819 | 0,012195 | 0,011688 | 0,006024 | 0,032275 | 0,017641 | 0,000031 |

O denominador de coverage foi 522.517 receitas elegíveis. Estes resultados são
somente dessa amostra; não representam automaticamente o desempenho no dataset
completo. `--max-users` seleciona usuários reprodutivelmente sem usar os
resultados das recomendações. A lista de IDs é salva em `selected_users.csv`, e os
arquivos de configuração e resultados registram usuários selecionados, avaliados e
excluídos, além dos motivos de exclusão. Use o mesmo `--random-state` para reproduzir
uma amostra e tamanhos diferentes para comparar escalas:

```shell
python -m src.evaluation.run_experiments --max-users 100 --random-state 42 --top-n 10 --output-dir results/report
```

Quando os recursos disponíveis permitirem, compare também uma amostra de 500:

```shell
python -m src.evaluation.run_experiments --max-users 500 --random-state 42 --top-n 10 --output-dir results/sample/500-users
```

Sem `--max-users`, o módulo processa todos os usuários; a execução completa não é
substituída automaticamente por uma amostra. Experimentos amostrais devem ser
identificados como amostras e não como desempenho no dataset completo.

A interface permite selecionar todos os usuários com pelo menos 50 interações;
esse critério é exibido no aplicativo e evita limitar a demonstração aos primeiros
IDs do dataset.

## Análise qualitativa

O módulo seleciona aleatoriamente, de forma reprodutível, até três usuários com
histórico de treino suficiente e avaliações positivas no teste, ou aceita dois ou
três IDs fornecidos. Para cada usuário registra o histórico positivo/negativo,
recomendações, pontuações, explicações e se itens relevantes do holdout ficaram fora
do Top-N. Limitações são derivadas desses resultados observados e exportadas em
`results/qualitative_examples.csv`.

## Tipos de testes realizados

- **Testes unitários:** verificam funções isoladas, incluindo ponderação das
  avaliações, construção do perfil, conversão de tempo, filtros, similaridade,
  métricas e explicações.
- **Testes de integração:** verificam o fluxo entre carregamento de dados, preparação
  textual, TF-IDF, perfil, recomendação e exportação dos resultados.
- **Experimentos offline:** medem a recuperação de receitas avaliadas positivamente
  no holdout, usando a divisão por usuário-receita, os baselines aleatório e de
  popularidade e as métricas descritas acima. Eles avaliam desempenho, não são testes
  automatizados.
- **Análise qualitativa:** inspeciona exemplos reais de usuários, seus históricos,
  recomendações e explicações.

Testes automatizados aprovados não significam que o algoritmo tenha bom desempenho
offline. Quantidade de testes e cobertura devem ser informadas somente após executar
os comandos de teste; resultados de experimentos dependem de execução real com os
Parquet disponíveis.

Na validação local de 12/11/2026, a suíte teve **43 testes aprovados** e cobertura
total de **81% em `src`** (`pytest-cov`). Esses valores descrevem os testes
automatizados, não a qualidade das recomendações.

## Limitações

- Perfis personalizados dependem de avaliações informativas no conjunto de treino.
- Itens sem termos reconhecidos pelo TF-IDF podem ter score nulo ou explicação geral.
- Receitas sem tempo verificável são excluídas se o filtro de tempo for usado.
- A qualidade das recomendações depende da qualidade dos textos e ingredientes do
  dataset.
- A avaliação completa percorre o catálogo por usuário e pode ser demorada; amostras
  de 100 e 500 usuários são alternativas reprodutíveis, mas não substituem a execução
  completa nem representam automaticamente todo o dataset.
- A amostra de 100 usuários foi executada; a amostra de 500 e a análise qualitativa
  completa não foram executadas nesta validação devido ao custo de processar o
  catálogo integral.
- Não declare qualidade subjetiva a partir apenas das métricas offline; a análise
  qualitativa auxilia a inspeção, mas não substitui avaliação com usuários.

## Tecnologias utilizadas

- Python: implementação.
- pandas e NumPy: preparação, agregações e métricas.
- scikit-learn: TF-IDF.
- SciPy: operações com matrizes esparsas.
- PyArrow: leitura de Parquet via pandas.
- Streamlit: interface demonstrativa.
- pytest e pytest-cov: testes automatizados e cobertura.

## Como obter o dataset

**Download manual:** acesse o [dataset no Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews),
baixe e extraia os arquivos Parquet.

**Kaggle CLI:** configure as credenciais do Kaggle e execute:

```shell
kaggle datasets download -d irkaal/foodcom-recipes-and-reviews -p data/raw --unzip
```

Confira os nomes dos arquivos extraídos. O projeto espera:

```text
data/raw/
├── recipes.parquet
└── reviews.parquet
```

Os arquivos brutos são ignorados pelo Git.

## Instalação

```shell
git clone https://github.com/Nevrand/filtragem-baseada-conteudo.git
cd filtragem-baseada-conteudo
python -m venv .venv
```

Ativar ambiente virtual:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Instalar dependências e preparar os Parquet em `data/raw/` conforme a seção anterior:

```shell
python -m pip install -r requirements.txt
```

## Comandos para análise e testes

Verificar os dados e exportar o relatório:

```shell
python -m src.analysis.dataset_report
```

Executar testes:

```shell
python -m pytest -v
python -m pytest --cov=src --cov-report=term-missing
```

Executar o experimento completo:

```shell
python -m src.evaluation.run_experiments
```

Executar experimento com parâmetros:

```shell
python -m src.evaluation.run_experiments --test-fraction 0.2 --random-state 42 --top-n 10
```

Executar a amostra recomendada de 100 usuários (não representa todo o dataset):

```shell
python -m src.evaluation.run_experiments --max-users 100 --random-state 42 --top-n 10 --output-dir results/report
```

Comparar com uma amostra de 500 usuários, se houver recursos suficientes:

```shell
python -m src.evaluation.run_experiments --max-users 500 --random-state 42 --top-n 10 --output-dir results/sample/500-users
```

Gerar análise qualitativa automaticamente ou para IDs existentes:

```shell
python -m src.evaluation.qualitative_analysis
python -m src.evaluation.qualitative_analysis --output results/report/qualitative_examples.csv
python -m src.evaluation.qualitative_analysis --users 123 456
```

Para IDs explícitos, substitua `123 456` por identificadores que existam no Parquet.

Executar a interface:

```shell
python -m streamlit run app/main.py
```

Arquivos locais em `results/` são ignorados por padrão. Somente arquivos finais
selecionados em `results/report/` são permitidos no Git; os Parquet nunca devem ser
versionados.

## Estrutura do projeto

```text
app/
  main.py                         Interface Streamlit
data/
  raw/                            Parquet local (não versionado)
results/
  sample/                         Resultados e arquivos locais ignorados pelo Git
  report/                         Resultados finais selecionados e versionáveis
src/
  analysis/
    dataset_report.py             Estatísticas descritivas do dataset
  data/
    load_data.py                  Leitura e validação dos Parquet
    preprocess.py                 Limpeza, duração e preparação textual
  evaluation/
    baselines.py                  Baselines aleatório e popularidade
    evaluate.py                   Métricas por usuário e resumo
    metrics.py                    Precision, Recall, nDCG e Coverage
    qualitative_analysis.py       Exemplos qualitativos reais
    run_experiments.py            CLI dos experimentos offline
    split.py                      Holdout por usuário
  features/
    text_features.py              Matriz TF-IDF esparsa
    user_profile.py               Perfis ponderados por nota
  recommendation/
    filters.py                    Filtros de candidatas
    recommender.py                Ranking, explicações e MMR
    similarity.py                 Similaridade de cosseno
tests/
  test_recipe_match.py            Testes unitários e de integração sintéticos
```

## Integrantes e contribuições

- **Integrante:** Alexander — **Contribuição:** Dados
- **Integrante:** Geovanna — **Contribuição:** Relatório e apresentação
- **Integrante:** Thercio — **Contribuição:** Frontend

## Referências

- Food.com Recipes and Reviews. Kaggle:
  <https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews>
- pandas: <https://pandas.pydata.org/>
- scikit-learn: <https://scikit-learn.org/>
- SciPy: <https://scipy.org/>
- Streamlit: <https://streamlit.io/>
