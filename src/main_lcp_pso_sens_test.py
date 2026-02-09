"""
Least Cost Path com PSO - Teste de Sensibilidade
Implementa teste de estabilidade com múltiplas seeds e estatística
"""

import numpy as np
import rasterio
import os
import sys
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append('..')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_lcp_pso import AdvancedPreprocessor
from cost_surface import CostSurface
from pso_optimizer import PSOOptimizer
from path_analysis import PathAnalyzer
from visualization import PathVisualizer
from raster_utils import coord_to_index


def main():
    # Iniciar timer global
    script_start_time = time.time()
    
    print("=" * 80)
    print("LEAST-COST PATH COM PSO - SOLUÇÃO ROBUSTA PARA TCC")
    print("Teste de Estabilidade (5 Seeds) + Anisotropia + Vetorização")
    print("=" * 80)
    
    # 1. Pipeline avançado de pré-processamento
    print("\n1. PIPELINE AVANÇADO DE PRÉ-PROCESSAMENTO")
    print("-" * 60)
    preprocess_start = time.time()
    
    preprocessor = AdvancedPreprocessor()
    
    data_paths = {
        'dem': "../data/raw/DEM.tif",
        'slope': "../data/raw/Declividade.tif", 
        'insol': "../data/raw/Insolação.tif",
        'visib': "../data/raw/Visibilidade.tif"
    }
    
    # Verifica se arquivos existem
    for name, path in data_paths.items():
        if not os.path.exists(path):
            print(f"ERRO: Arquivo não encontrado: {path}")
            return

    processed_data = preprocessor.apply_full_pipeline(
        data_paths['dem'], data_paths['slope'], 
        data_paths['insol'], data_paths['visib']
    )
    
    preprocess_elapsed = time.time() - preprocess_start
    print(f"   Pré-processamento concluído em {preprocess_elapsed:.2f}s")
    
    # Coordenadas
    start_coords = (-37.3076, -8.4850)
    goal_coords = (-37.23920, -8.50260)
    
    # 2. Inicializar componentes
    print("\n2. INICIALIZANDO COMPONENTES AVANÇADOS")
    print("-" * 60)
    
    # --- CALIBRAÇÃO FINA ---
    cost_surface_gen = CostSurface(
        processed_data['slope_norm'],
        processed_data['visib_norm'],
        processed_data['insol_norm'],
        normalization_method='rank_uniform',
        base_friction=0.1 
    )
    
    path_analyzer = PathAnalyzer(processed_data['transform'], processed_data['shape'])
    
    with rasterio.open(data_paths['dem']) as src:
        dem = src.read(1).astype(float)
        dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
        transform = src.transform
        bounds = src.bounds
    
    visualizer = PathVisualizer(dem, transform, bounds)
    
    # --- CONFIGURAÇÃO DO TESTE DE ROBUSTEZ ---
    regularization_strength = 3.0
    seeds_para_teste = [42, 100, 2026, 7, 999] # 5 Seeds para validar estabilidade
    
    print("   Componentes inicializados!")
    print(f"   Modo de Teste: 5 Rodadas (Seeds: {seeds_para_teste})")
    print(f"   Configuração PSO: 40 partículas, 30 iterações (Exploração Aumentada)")
    print(f"   Regularização: {regularization_strength}")
    
    # 3. Executar Loop de Robustez
    print("\n3. EXECUTANDO TESTE DE ROBUSTEZ (MULTIPLE SEEDS)")
    print("-" * 60)
    
    resultados_pesos = []
    scores_finais = []
    
    # Variáveis para guardar o MELHOR resultado de todos para gerar os mapas
    best_overall_results = None
    best_overall_score = float('inf')
    best_overall_seed = -1
    
    loop_start = time.time()
    
    for seed in seeds_para_teste:
        print(f"\n>>> Rodando Seed {seed}...")
        
        pso = PSOOptimizer(n_particles=40, n_iterations=30, seed=seed)
        
        res = pso.optimize(
            path_analyzer=path_analyzer,
            cost_surface_generator=cost_surface_gen,
            start_coords=start_coords,
            goal_coords=goal_coords,
            alpha=1.0,
            sum_penalty_weight=10.0,
            verbose=False, 
            use_grass_mode=True,
            dem=dem,
            parallel=True, # Tente True se tiver multicore, senão False
            regularization_strength=regularization_strength
        )
        
        # Coletar dados
        w = res['best_weights']
        s = res['best_score']
        
        resultados_pesos.append(w)
        scores_finais.append(s)
        
        print(f"   [Seed {seed}] Pesos: {np.round(w, 4)} | Score: {s:.4f}")
        
        # Verificar se é o melhor absoluto
        if s < best_overall_score:
            best_overall_score = s
            best_overall_results = res
            best_overall_seed = seed
            
    loop_elapsed = time.time() - loop_start
    print(f"\n   Teste de robustez concluído em {loop_elapsed:.2f}s")
    
    # 4. Análise Estatística
    print("\n4. ANÁLISE ESTATÍSTICA E CONSOLIDAÇÃO")
    print("-" * 60)
    
    pesos_array = np.array(resultados_pesos)
    media_pesos = np.mean(pesos_array, axis=0)
    desvio_pesos = np.std(pesos_array, axis=0)
    mean_std = np.mean(desvio_pesos)
    
    print(f"   MÉDIA DOS PESOS (Slope, Visib, Insol):")
    print(f"   {np.round(media_pesos, 4)}")
    print(f"\n   DESVIO PADRÃO (+/-):")
    print(f"   {np.round(desvio_pesos, 4)}")
    
    estabilidade = "ALTA" if mean_std < 0.05 else "MÉDIA/BAIXA"
    print(f"\n   Diagnóstico de Estabilidade: {estabilidade} (Mean Std: {mean_std:.4f})")
    
    # Usar o melhor resultado para os arquivos finais
    print(f"\n   Selecionando melhor resultado (Seed {best_overall_seed}) para geração de mapas...")
    results = best_overall_results
    best_weights = results['best_weights']
    best_metrics = results['best_metrics']
    
    # 5. Salvamento dos Resultados
    results_final = {
        'method': 'PSO_ROBUST_TEST_5SEEDS',
        'statistics': {
            'mean_weights': list(media_pesos),
            'std_weights': list(desvio_pesos),
            'seeds_used': seeds_para_teste,
            'best_seed': best_overall_seed
        },
        'best_weights': {
            'slope': float(best_weights[0]),
            'visib': float(best_weights[1]),
            'insol': float(best_weights[2])
        },
        'best_metrics': {
            'cost': float(best_metrics['cost']),
            'length': float(best_metrics['length']),
            'curvature': float(best_metrics['curvature']),
            'efficiency': float(best_metrics['efficiency'])
        }
    }
    
    os.makedirs('results_final', exist_ok=True)
    with open('results_final/main_lcp_pso_robust_results.json', 'w') as f:
        json.dump(results_final, f, indent=2)
        
    # Gerar superfície final para visualização
    final_cost_surface = cost_surface_gen.generate_cost_surface(best_weights)
    output_cost_path = 'results_final/final_cost_surface.tif'
    
    # Salvar TIF
    with rasterio.open(data_paths['dem']) as dem_src:
        profile = dem_src.profile
        profile.update({'dtype': 'float32', 'count': 1, 'compress': 'lzw'})
        with rasterio.open(output_cost_path, 'w', **profile) as dst:
            dst.write(final_cost_surface.astype(np.float32), 1)
            
    # Plotar visualizações
    try:
        # Convergência (do melhor run)
        visualizer.plot_convergence(results['convergence_history'], save_path='results_final/pso_convergence_best.png')
        
        # Superfícies Individuais
        if hasattr(visualizer, 'plot_individual_cost_surfaces'):
            # Mudar para o diretório results_final antes de plotar
            original_dir = os.getcwd()
            os.chdir('results_final')
            visualizer.plot_individual_cost_surfaces(
                cost_surface_gen, 
                best_weights
            )
            os.chdir(original_dir)  # Voltar ao diretório original
            
            # Gerar superfície de custo total
            visualizer.plot_total_cost_surface(
                cost_surface_gen, 
                best_weights, 
                'sens_test'
            )
            
        plt.figure(figsize=(12, 8))
 
        im = plt.imshow(final_cost_surface, cmap='viridis_r', origin='upper') 
        
        cbar = plt.colorbar(im)
        cbar.set_label('Custo (Claro=Barato, Escuro=Caro)', rotation=270, labelpad=15)
        
        if best_metrics['path']:
            opt_path = np.array(best_metrics['path'])
            plt.plot(opt_path[:, 1], opt_path[:, 0], 'r-', linewidth=2, label='LCP Otimizado')
            plt.legend()
            
        plt.title(f"Melhor Caminho LCP (Seed {best_overall_seed})\nPesos: {np.round(best_weights, 3)}")
        plt.savefig('results_final/lcp_path_on_cost_surface.png', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Erro ao gerar plots: {e}")
        import traceback
        traceback.print_exc()

    print("\nPIPELINE FINAL CONCLUÍDO COM SUCESSO!")
    print(f"Tempo total de execução: {time.time() - script_start_time:.2f} segundos")

if __name__ == '__main__':
    main()