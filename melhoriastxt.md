# TASKS
### * Corrigir fórmula da regularização
### * Comparar com outros modelos(Arima, Zhang, Pranighari, ELM construtivo).
### * Discutir resultados
### EM ANDAMENTO Melhorar relatório(rsl, referencias, discutir trabalhos)
### * qr fatorization Incremental ELM
### EM ANDAMENTO implementar IELM
### * Buscar mais trabalhos
### - adicionar resultados do IELM ao relatório
### * Combinar ETS com CC e ELM
### * Combinar ARIMA com ELM
### - ler trabalhos encontrados
### - Melhorar Estado da Arte Cascade
### - Melhorar Estado da Arte Séries Temporais
### - Melhorar Estado da ELM
### * [Implementar](https://link.springer.com/article/10.1007/s11063-020-10294-9)
### * [Ler](https://www.sciencedirect.com/science/article/abs/pii/S0950705119301327?via%3Dihub)
### EM ANDAMENTO investigar Arima-Cascade, para descorbrir por que os resultados estão inferiores.
### - analisar comportamento do erro após a adicão de cada neurônio ELM construtivo nos residuos.
### - melhorar argumentação e motivação do relatório.
### EM ANDAMENTO descrever os métodos utilizados nos experimentos, assim como os parâmetros.
### * searchgrid svr c epsilon e gama
### - mlpets, svrets, ets.
### - combinar svr arima como entrada para outras redes
### EM ANDAMENTO descrever as bases com uma tabela. quantos pontos de cada uma
### - tendencia, sazonalidade.
### - utlizar outras metricas.
### - mape é assimétrica.
### - utilizar PSO para combinar.
### - salvar configuração selecionada (dar preferencia ao rbf).
### - optuna.
 
# Alterar relatório
## INTRODUÇÃO
### - o foco não é a velocidade de execução.
### - melhorar discussão sobre dificulddes de ANN na previsão de residuos(ajuste do modelo e a previsão em).
### - Mudar estilo da citação (autor, ano)
### - Nos objetivos Retormar rapidamente a motivação

## PROPOSTA
### * Corrigir fórmula da regularização
### * comentar cada etapa do processo de previsão
### * No desenho, trocar ARIMA por modelagem linear e ccelm por modelagem não-linear
### - adicionar critério  de parada após o erro não diminuir após dez execuções.
### - melhorar explicação do ELM construtivo
### * testar elm-construtivo no residuo do arima.
### - testar mlp regressor
### * combinar ets com modelos não-lineares(CC, SVR, ELM).
