# LCP-PSO: Least Cost Path with Particle Swarm Optimization

Implementação de otimização de caminhos de custo mínimo usando PSO para calibração automática de pesos em análise arqueológica.

## Contexto

Este projeto implementa uma metodologia para encontrar caminhos ótimos em paisagens arqueológicas do Parque Nacional do Catimbau (PE), Brasil. O algoritmo PSO calibra automaticamente os pesos de três variáveis ambientais (declividade, visibilidade e insolação) para minimizar o custo de deslocamento.

## Estrutura do Projeto

```
src/
├── main_lcp_pso_final.py          # Script principal de execução
├── main_lcp_pso_sens_test.py      # Teste de robustez com múltiplas 
├── pipeline_lcp_pso.py            # Pré-processamento de dados
├── cost_surface.py                # Geração de superfícies de custo
├── pso_optimizer.py               # Otimizador PSO
├── path_analysis.py               # Análise de caminhos
├── visualization.py               # Geração de visualizações
└── raster_utils.py                # Utilitários para dados raster
```

## Dependências

- Python 3.8+
- numpy
- rasterio
- matplotlib
- scipy
- multiprocessing (padrão Python)

## Como Executar

### 1. Preparação dos Dados

Coloque os arquivos raster na pasta `data/raw/`:
- `DEM.tif` - Modelo Digital de Elevação
- `Declividade.tif` - Camada de declividade
- `Insolacao.tif` - Camada de insolação
- `Visibilidade.tif` - Camada de visibilidade

### 2. Execução Principal

```bash
cd src
python main_lcp_pso_final.py
```

### 3. Teste de Sensibilidade

```bash
cd src
python main_lcp_pso_sens_test.py
```

## Saídas

Os resultados são salvos na pasta `results_final/`:
- Superfícies de custo individuais (PNG)
- Superfície de custo total (PNG)
- Gráficos de convergência (PNG)
- Perfis altimétricos (PNG)
- Logs de execução (TXT)
- Resumo comparativo (TXT)
- Resultados detalhados (JSON)

## Metodologia

1. **Pré-processamento**: Transformação e normalização por postos das variáveis
2. **Otimização PSO**: Calibração automática de pesos com regularização entrópica
3. **Análise de Caminhos**: Grafo anisotrópico com 16 vizinhos
4. **Validação**: Protocolo de robustez com 5 sementes aleatórias

## Parâmetros Principais

- PSO: 15 partículas, 30 iterações (execução principal)
- PSO: 40 partículas, 30 iterações (teste de sensibilidade)
- Regularização: λ = 3.0 (divergência Kullback-Leibler)
- Sementes: {42, 100, 2026, 7, 999}

## Autor

Gabriela Maia Rodrigues
Trabalho de Conclusão de Curso - 2026