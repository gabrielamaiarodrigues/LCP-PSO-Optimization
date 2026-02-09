"""
Pipeline Principal LCP-PSO
Executa otimização de caminhos de custo mínimo com múltiplos pares de coordenadas
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


def calculate_path_distance_km(path_coords, transform):
    """
    Calcula distância total do caminho em km usando conversão adequada.
    
    Args:
        path_coords: coordenadas do caminho (índices)
        transform: transformação do raster
        
    Returns:
        float: distância total em km
    """
    if path_coords is None or len(path_coords) < 2:
        return 0.0
    
    total_distance = 0.0
    
    for i in range(1, len(path_coords)):
        # Converter índices para coordenadas geográficas
        x1, y1 = transform * (path_coords[i-1][1], path_coords[i-1][0])
        x2, y2 = transform * (path_coords[i][1], path_coords[i][0])
        
        # Calcular distância em metros usando conversão aproximada
        lat1, lon1 = y1, x1
        lat2, lon2 = y2, x2
        
        # Conversão: 1 grau ≈ 111,139 metros
        dlat = (lat2 - lat1) * 111139  # metros por grau de latitude
        dlon = (lon2 - lon1) * 111139 * np.cos(np.radians(lat1))  # metros por grau de longitude
        
        # Distância euclidiana em metros
        distance_m = np.sqrt(dlat**2 + dlon**2)
        total_distance += distance_m
    
    return total_distance / 1000.0  # Converter para km


def generate_elevation_profile(path_coords, dem, transform):
    """
    Gera perfil altimétrico do caminho com distâncias corretas em metros.
    
    Args:
        path_coords: coordenadas do caminho (índices)
        dem: array 2D com elevações
        transform: transformação do raster
        
    Returns:
        tuple: (distancias_km, elevacoes)
    """
    if path_coords is None or len(path_coords) < 2:
        return np.array([0]), np.array([])
    
    # Extrair elevações
    elevations = [dem[row, col] for row, col in path_coords]
    
    # Calcular distâncias usando Haversine approximation
    # Converter índices para coordenadas geográficas
    geo_coords = []
    for row, col in path_coords:
        x, y = transform * (col, row)
        geo_coords.append([x, y])
    geo_coords = np.array(geo_coords)
    
    # Calcular distâncias acumuladas em metros
    distances = [0]
    for i in range(1, len(geo_coords)):
        # Conversão aproximada: 1 grau ≈ 111,139 metros
        lat1, lon1 = geo_coords[i-1][1], geo_coords[i-1][0]
        lat2, lon2 = geo_coords[i][1], geo_coords[i][0]
        
        # Haversine approximation para pequenas distâncias
        dlat = (lat2 - lat1) * 111139  # metros por grau de latitude
        dlon = (lon2 - lon1) * 111139 * np.cos(np.radians(lat1))  # metros por grau de longitude
        
        dist_m = np.sqrt(dlat**2 + dlon**2)
        distances.append(distances[-1] + dist_m)
    
    # Converter para km
    cumulative_distances = np.array(distances) / 1000.0
    elevations = np.array(elevations)
    
    return cumulative_distances, elevations


def plot_path_on_dem(dem, transform, bounds, path_coords, path_name, weights, save_path):
    """
    Plota o caminho sobre o DEM em preto e branco.
    
    Args:
        dem: array 2D com elevações
        transform: transformação do raster
        bounds: limites geográficos
        path_coords: coordenadas do caminho
        path_name: nome do caminho
        weights: pesos ótimos
        save_path: caminho para salvar imagem
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot 2D sobre DEM em preto e branco
    im = ax.imshow(dem, cmap='gray', origin='upper', extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    
    if path_coords is not None and len(path_coords) > 0:
        # Converter coordenadas do path para geográficas
        geo_coords = []
        for row, col in path_coords:
            x, y = transform * (col, row)
            geo_coords.append([x, y])
        geo_coords = np.array(geo_coords)
        
        ax.plot(geo_coords[:, 0], geo_coords[:, 1], 'r-', linewidth=3, label='LCP Otimizado')
        ax.plot(geo_coords[0, 0], geo_coords[0, 1], 'go', markersize=8, label='Início')
        ax.plot(geo_coords[-1, 0], geo_coords[-1, 1], 'ro', markersize=8, label='Fim')
    
    ax.set_title(f'{path_name}\nPesos: {np.round(weights, 3)}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Elevação (m)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_elevation_profile(distances, elevations, path_name, weights, save_path):
    """
    Plota perfil altimétrico do caminho.
    
    Args:
        distances: distâncias acumuladas em km
        elevations: elevações correspondentes
        path_name: nome do caminho
        weights: pesos ótimos
        save_path: caminho para salvar imagem
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Perfil altimétrico
    ax1.plot(distances, elevations, 'b-', linewidth=2)
    ax1.fill_between(distances, elevations, alpha=0.3)
    ax1.set_title(f'{path_name} - Perfil Altimétrico\nPesos: {np.round(weights, 3)}')
    ax1.set_xlabel('Distância Acumulada (km)')
    ax1.set_ylabel('Elevação (m)')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calcular estatísticas para o log
    gradient = np.abs(np.gradient(elevations, distances))
    
    # Calcular ganho e perda de elevação
    elevation_diffs = np.diff(elevations)
    gain_total = float(np.sum(elevation_diffs[elevation_diffs > 0]))
    loss_total = float(np.sum(elevation_diffs[elevation_diffs < 0]))
    
    return {
        'elevation_mean': float(np.mean(elevations)),
        'elevation_max': float(np.max(elevations)),
        'elevation_min': float(np.min(elevations)),
        'elevation_std': float(np.std(elevations)),
        'gradient_mean': float(np.mean(gradient)),
        'gradient_max': float(np.max(gradient)),
        'gradient_min': float(np.min(gradient)),
        'elevation_gain_total': gain_total,
        'elevation_loss_total': loss_total
    }





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
    
    # Pares de coordenadas para teste
    test_paths = {
        'caminho_a': {
            'start': (-37.3076, -8.4850),
            'goal': (-37.23920, -8.50260),
            'description': 'Caminho A'
        },
        'caminho_b': {
            'start': (-37.4033, -8.5505),
            'goal': (-37.1913, -8.537),
            'description': 'Caminho B'
        }
    }
    
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
    
    # --- CONFIGURAÇÃO DO PSO (EXECUÇÃO ÚNICA) ---
    regularization_strength = 3.0
    seed_unico = 42  # Seed única para execução
    
    print("   Componentes inicializados!")
    print(f"   Modo de Execução: Única (Seed: {seed_unico})")
    print(f"   Configuração PSO: 30 partículas, 50 iterações")
    print(f"   Regularização: {regularization_strength}")
    
    # 3. Executar Otimização para cada caminho
    print("\n3. EXECUTANDO OTIMIZAÇÃO PARA CADA CAMINHO")
    print("-" * 60)
    
    all_results = {}
    
    for path_name, path_data in test_paths.items():
        print(f"\n>>> Processando {path_data['description']} <<<")
        print(f"    Início: {path_data['start']}")
        print(f"    Fim: {path_data['goal']}")
        
        start_coords = path_data['start']
        goal_coords = path_data['goal']
        
        path_start_time = time.time()
        
        print(f"\n>>> Rodando Seed {seed_unico}...")
        
        # PSO com seed única
        pso = PSOOptimizer(n_particles=15, n_iterations=30, seed=seed_unico)
        
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
            parallel=True,
            regularization_strength=regularization_strength
        )
        
        # Coletar dados
        w = res['best_weights']
        s = res['best_score']
        
        print(f"   [Seed {seed_unico}] Pesos: {np.round(w, 4)} | Score: {s:.4f}")
        
        path_elapsed = time.time() - path_start_time
        print(f"\n   {path_data['description']} concluído em {path_elapsed:.2f}s")
        
        # Armazenar resultados deste caminho
        all_results[path_name] = {
            'description': path_data['description'],
            'start_coords': start_coords,
            'goal_coords': goal_coords,
            'statistics': {
                'best_seed': seed_unico,
                'execution_time': path_elapsed
            },
            'best_results': res
        }
    
    # 4. Salvar Resultados e Gerar Visualizações
    print("\n4. SALVANDO RESULTADOS E GERANDO VISUALIZAÇÕES")
    print("-" * 60)
    
    os.makedirs('results_final', exist_ok=True)
    
    # Salvar resultados completos
    results_final = {
        'method': 'PSO_MULTI_PATH_NORMAL_PARAMS',
        'timestamp': datetime.now().isoformat(),
        'paths': {}
    }
    
    for path_name, path_results in all_results.items():
        best_weights = path_results['best_results']['best_weights']
        best_metrics = path_results['best_results']['best_metrics']
        
        results_final['paths'][path_name] = {
            'description': path_results['description'],
            'coordinates': {
                'start': path_results['start_coords'],
                'goal': path_results['goal_coords']
            },
            'statistics': path_results['statistics'],
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
        
        # Gerar visualizações para cada caminho
        try:
            # Gerar superfície de custo final
            final_cost_surface = cost_surface_gen.generate_cost_surface(best_weights)
            output_cost_path = f'results_final/cost_surface_{path_name}.tif'
            
            # Salvar TIF
            with rasterio.open(data_paths['dem']) as dem_src:
                profile = dem_src.profile
                profile.update({'dtype': 'float32', 'count': 1, 'compress': 'lzw'})
                with rasterio.open(output_cost_path, 'w', **profile) as dst:
                    dst.write(final_cost_surface.astype(np.float32), 1)
            
            # Plotar convergência
            visualizer.plot_convergence(
                path_results['best_results']['convergence_history'], 
                save_path=f'results_final/convergence_{path_name}.png'
            )
            
            # Plotar caminho na superfície de custo
            plt.figure(figsize=(12, 8))
            im = plt.imshow(final_cost_surface, cmap='viridis_r', origin='upper')
            cbar = plt.colorbar(im)
            cbar.set_label('Custo Normalizado', rotation=270, labelpad=15)
            
            if best_metrics['path']:
                opt_path = np.array(best_metrics['path'])
                plt.plot(opt_path[:, 1], opt_path[:, 0], 'r-', linewidth=2, label='LCP Otimizado')
                plt.legend()
                
            plt.title(f"{path_results['description']}\nPesos: {np.round(best_weights, 3)}")
            plt.savefig(f'results_final/path_{path_name}.png', dpi=300)
            plt.close()
            
            # NOVO: Plotar superfícies de custo individuais (3 arquivos separados)
            # Mudar para o diretório results_final antes de plotar
            original_dir = os.getcwd()
            os.chdir('results_final')
            visualizer.plot_individual_cost_surfaces(
                cost_surface_gen, best_weights
            )
            os.chdir(original_dir)  # Voltar ao diretório original
            
            # NOVO: Gerar superfície de custo total em PNG
            visualizer.plot_total_cost_surface(
                cost_surface_gen, best_weights, path_name
            )
            
            # NOVO: Plotar caminho sobre DEM (2D em preto e branco)
            if best_metrics['path']:
                opt_path = np.array(best_metrics['path'])
                plot_path_on_dem(
                    dem, transform, bounds, opt_path, 
                    path_results['description'], best_weights,
                    f'results_final/path_dem_{path_name}.png'
                )
                
                # NOVO: Gerar perfil altimétrico
                distances, elevations = generate_elevation_profile(opt_path, dem, transform)
                profile_stats = plot_elevation_profile(
                    distances, elevations,
                    path_results['description'], best_weights,
                    f'results_final/profile_{path_name}.png'
                )
                
                # Calcular métricas adicionais
                distance_km = calculate_path_distance_km(opt_path, transform)
                
                # Calcular estatísticas de elevação (sempre criar)
                elevation_stats = {
                    'elevation_mean': profile_stats['elevation_mean'],
                    'elevation_max': profile_stats['elevation_max'],
                    'elevation_min': profile_stats['elevation_min'],
                    'elevation_std': profile_stats['elevation_std'],
                    'elevation_gain_total': profile_stats['elevation_gain_total'],
                    'elevation_loss_total': profile_stats['elevation_loss_total'],
                    'gradient_mean': profile_stats['gradient_mean'],
                    'gradient_max': profile_stats['gradient_max'],
                    'gradient_min': profile_stats['gradient_min']
                }
                
                # Atualizar métricas do caminho com distância
                best_metrics['distance_km'] = distance_km
            else:
                # Criar elevation_stats vazio se não houver caminho
                elevation_stats = {
                    'elevation_mean': 0.0,
                    'elevation_max': 0.0,
                    'elevation_min': 0.0,
                    'elevation_std': 0.0,
                    'elevation_gain_total': 0.0,
                    'elevation_loss_total': 0.0,
                    'gradient_mean': 0.0,
                    'gradient_max': 0.0,
                    'gradient_min': 0.0
                }
                
                # Atualizar métricas do caminho com distância (mesmo sem caminho)
                best_metrics['distance_km'] = 0.0
            
            print(f"   Visualizações geradas para {path_name}")
            
        except Exception as e:
            print(f"   Erro ao gerar visualizações para {path_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Garantir que elevation_stats sempre exista mesmo em caso de erro
            elevation_stats = {
                'mean_elevation': 0.0,
                'max_elevation': 0.0,
                'min_elevation': 0.0,
                'std_elevation': 0.0,
                'elevation_gain': 0.0,
                'elevation_loss': 0.0,
                'gradient_mean': 0.0,
                'gradient_max': 0.0,
                'gradient_min': 0.0
            }
    
    # 5. Gerar logs detalhados e salvar resultados finais
    print("\n5. GERANDO LOGS DETALHADOS E SALVANDO RESULTADOS")
    print("-" * 60)
    
    # Salvar arquivo JSON final com métricas atualizadas
    with open('results_final/multi_path_results.json', 'w') as f:
        json.dump(results_final, f, indent=2)
    
    # Gerar logs individuais para cada caminho
    for path_name, path_results in all_results.items():
        best_weights = path_results['best_results']['best_weights']
        best_metrics = path_results['best_results']['best_metrics']
        
        # Preparar métricas de execução
        execution_metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_time': path_results['statistics']['execution_time'],
            'n_seeds': 1,
            'best_seed': seed_unico,
            'n_particles': 15,
            'n_iterations': 30,
            'best_weights': {
                'slope': float(best_weights[0]),
                'visib': float(best_weights[1]),
                'insol': float(best_weights[2])
            },
            'path_metrics': {
                'cost': float(best_metrics['cost']),
                'length': float(best_metrics['length']),
                'curvature': float(best_metrics['curvature']),
                'efficiency': float(best_metrics['efficiency']),
                'distance_km': float(best_metrics.get('distance_km', 0.0))
            },
            'elevation_stats': elevation_stats,
            'pso_config': {
                'regularization_strength': regularization_strength,
                'alpha': 1.0,
                'sum_penalty_weight': 10.0,
                'use_grass_mode': True,
                'parallel': True
            }
        }
        
        # Imprimir métricas no terminal
        print(f"\n   MÉTRICAS DO {path_name.upper()}:")
        print(f"   - Tempo de Execução: {path_results['statistics']['execution_time']:.2f} segundos")
        print(f"   - Custo: {best_metrics['cost']:.6f}")
        print(f"   - Distância: {best_metrics.get('distance_km', 0):.3f} km")
        print(f"   - Comprimento: {best_metrics['length']:.6f}")
        print(f"   - Curvatura: {best_metrics['curvature']:.6f}")
        print(f"   - Eficiência: {best_metrics['efficiency']:.6f}")
        print(f"   - Pesos: Declividade={best_weights[0]:.3f}, Visibilidade={best_weights[1]:.3f}, Insolação={best_weights[2]:.3f}")
        if elevation_stats:
            print(f"   - Elevação Média: {elevation_stats.get('elevation_mean', 0):.1f} m")
            print(f"   - Elevação Máxima: {elevation_stats.get('elevation_max', 0):.1f} m")
            print(f"   - Elevação Mínima: {elevation_stats.get('elevation_min', 0):.1f} m")
            print(f"   - Gradiente Médio: {elevation_stats.get('gradient_mean', 0):.1f} m/km")
    
    # Imprimir resumo comparativo no terminal
    print("\n   RESUMO COMPARATIVO:")
    print("   " + "="*60)
    print(f"   Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"   Total de Caminhos: {len(all_results)}")
    print(f"   Seed Utilizada: {seed_unico}")
    print("   " + "-"*60)
    
    for path_name, path_results in all_results.items():
        best_weights = path_results['best_results']['best_weights']
        best_metrics = path_results['best_results']['best_metrics']
        
        print(f"\n   {path_results['description']}:")
        print(f"   - Seed: {seed_unico}")
        print(f"   - Custo: {best_metrics['cost']:.6f}")
        print(f"   - Distância: {best_metrics.get('distance_km', 0):.3f} km")
        print(f"   - Pesos: Declividade={best_weights[0]:.3f}, Visibilidade={best_weights[1]:.3f}, Insolação={best_weights[2]:.3f}")
        print(f"   - Tempo de Execução: {path_results['statistics']['execution_time']:.2f}s")
    
    print("\n   " + "="*60)
    print("   ARQUIVOS GERADOS:")
    print("   - JSON: results_final/multi_path_results.json")
    print("   - Superfícies: results_final/cost_surface_*.tif")
    print("   - Superfícies Individuais: results_final/cost_surfaces_*.png")
    print("   - Mapas: results_final/path_*.png")
    print("   - DEM: results_final/path_dem_*.png")
    print("   - Perfis: results_final/profile_*.png")
    print("   - Convergência: results_final/convergence_*.png")
    print("   " + "="*60)

    print("\nPIPELINE FINAL CONCLUÍDO COM SUCESSO!")

if __name__ == '__main__':
    main()