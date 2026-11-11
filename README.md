# RecipeMatch

Trabalho bimestral de Oficina de Desenvolvimento de Sistemas I.

## Sobre o projeto

RecipeMatch recomenda receitas com base no conteúdo das receitas avaliadas por cada
usuário. O sistema representa nomes, descrições, categorias, palavras-chave e
ingredientes com TF-IDF e compara o perfil aprendido do usuário com as receitas do
catálogo por similaridade de cosseno.

Recomendação por conteúdo personaliza a lista usando características das receitas do
histórico individual. Recomendação por popularidade, usada como baseline, ordena os
itens pelo número de avaliações no conjunto de treinamento e não personaliza por
preferências textuais.

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

Receitas fornecem os textos e atributos usados no TF-IDF. Avaliações ligam usuários
às receitas e fornecem as notas para construir perfis e o holdout offline. O dataset
observado possui notas de 0 a 5; a nota 0 é tratada como a avaliação mais negativa
no perfil. Os Parquet não são versionados por seu tamanho; obtenha-os no Kaggle e
coloque-os em `data/raw/`.

O comando `dataset_report.py` recalcula esses valores a partir dos arquivos locais.
Na execução real local de 11/11/2026, o relatório calculou 522.517 receitas,
1.401.982 avaliações, 271.907 usuários distintos e 271.678 receitas avaliadas.
Foram encontradas 520.298 receitas com ingredientes. As notas observadas variaram
de 0 a 5, com média 4,40795. Esses números descrevem somente os Parquet disponíveis
nessa execução e podem mudar em outra versão do dataset.

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

Os pesos implementados são:

| Nota | Peso | Interpretação |
|---:|---:|---|
| 5 | +2 | positiva forte |
| 4 | +1 | positiva |
| 3 | 0 | neutra |
| 2 | -1 | negativa |
| 1 | -2 | negativa forte |
| 0 | -2 | valor presente no arquivo observado; tratado como menor nota |

Avaliações repetidas do mesmo usuário para a mesma receita são agregadas pela média
antes de aplicar o peso. Isso evita que uma receita repetida conte várias vezes; uma
média igual a 3 não contribui ao perfil.

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

O protocolo separa eventos por usuário. Se todas as datas forem válidas, as avaliações
mais recentes ficam no teste; caso contrário, usa divisão aleatória por usuário com
semente fixa. Usuários precisam ter avaliações suficientes no treino para formar um
perfil. O perfil usa exclusivamente o treino; receitas vistas no treino são removidas
das candidatas, enquanto as receitas do teste permanecem elegíveis. Uma avaliação
4 ou 5 no teste define item relevante. Usuários sem positivos no teste ou sem perfil
informativo aparecem nos resultados individuais com o motivo e não entram nas médias.

Métricas:

- **Precision@K:** relevantes recomendados nos primeiros K, divididos por K.
- **Recall@K:** relevantes recomendados nos primeiros K, divididos pelo total de
  itens relevantes no teste.
- **nDCG@K:** ganho binário descontado pelo logaritmo da posição, normalizado pelo
  ranking ideal.
- **Coverage:** receitas distintas recomendadas divididas pelo total de receitas
  elegíveis, definido como a união das candidatas dos usuários avaliáveis após
  excluir receitas vistas no treino; esse denominador é exportado em
  `metrics_summary.csv`.

## Experimentos e resultados

O avaliador compara RecipeMatch TF-IDF, recomendação aleatória com semente fixa e
popularidade calculada exclusivamente nas avaliações de treino. Todos recebem os
mesmos usuários e, para cada usuário, as mesmas candidatas após a exclusão do
histórico de treino.

| Modelo | Precision@5 | Recall@5 | nDCG@5 | Precision@10 | Recall@10 | nDCG@10 | Coverage (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RecipeMatch-TF-IDF | 0 | 0 | 0 | 0 | 0 | 0 | 0,0191 |
| Aleatório | 0 | 0 | 0 | 0 | 0 | 0 | 0,0191 |
| Popularidade | 0 | 0 | 0 | 0 | 0 | 0 | 0,0021 |

Esta execução piloto real usou 10 usuários amostrados com semente 42, divisão
cronológica, e Top-10; todos os três modelos tiveram métricas de ranking iguais a
zero nesta amostra. Coverage é mostrada como percentual. O resultado é limitado e
não representa desempenho no dataset completo. Para reproduzi-lo:

```shell
python -m src.evaluation.run_experiments --max-users 10 --random-state 42 --top-n 10 --output-dir results/sample
```

`--max-users` permite uma amostra aleatória reprodutível para testes rápidos. Sem
esse parâmetro, o módulo processa todos os usuários; resultados amostrados não devem
ser apresentados como avaliação do dataset completo.

A interface permite selecionar todos os usuários com pelo menos 50 avaliações;
esse critério é exibido no aplicativo e evita limitar a demonstração aos primeiros
IDs do dataset.

## Análise qualitativa

O módulo seleciona aleatoriamente, de forma reprodutível, até três usuários com
histórico de treino suficiente e avaliações positivas no teste, ou aceita dois ou
três IDs fornecidos. Para cada usuário registra o histórico positivo/negativo,
recomendações, pontuações, explicações e se itens relevantes do holdout ficaram fora
do Top-N. Limitações são derivadas desses resultados observados e exportadas em
`results/qualitative_examples.csv`.

## Limitações

- Perfis personalizados dependem de avaliações informativas no conjunto de treino.
- Itens sem termos reconhecidos pelo TF-IDF podem ter score nulo ou explicação geral.
- Receitas sem tempo verificável são excluídas se o filtro de tempo for usado.
- A qualidade das recomendações depende da qualidade dos textos e ingredientes do
  dataset.
- A avaliação completa percorre o catálogo por usuário e pode ser demorada; use
  `--max-users` apenas para uma amostra explicitamente identificada.
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

```shell
# Linux/macOS
source .venv/bin/activate
```

Instalar dependências e preparar os Parquet em `data/raw/` conforme a seção anterior:

```shell
python -m pip install -r requirements.txt
```

## Comandos para análise e testes

Verificar os dados:

```shell
python -m src.analysis.dataset_report
```

Executar testes:

```shell
python -m pytest -v
python -m pytest --cov=src --cov-report=term-missing
```

Executar experimentos completos:

```shell
python -m src.evaluation.run_experiments
```

Executar experimento com parâmetros:

```shell
python -m src.evaluation.run_experiments --test-fraction 0.2 --random-state 42 --top-n 10
```

Teste rápido com uma amostra reprodutível de usuários:

```shell
python -m src.evaluation.run_experiments --max-users 100 --output-dir results/sample
```

Gerar análise qualitativa automaticamente ou para IDs existentes:

```shell
python -m src.evaluation.qualitative_analysis
python -m src.evaluation.qualitative_analysis --users 123 456
```

Para IDs explícitos, substitua `123 456` por identificadores que existam no Parquet.

Executar a interface:

```shell
python -m streamlit run app/main.py
```

Os relatórios e experimentos geram arquivos CSV/JSON em `results/`, pasta local
ignorada pelo Git.

## Estrutura do projeto

```text
app/
  main.py                         Interface Streamlit
data/
  raw/                            Parquet local (não versionado)
results/                          CSV/JSON locais (não versionados)
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
