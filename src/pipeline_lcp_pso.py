"""
Pipeline de Pré-processamento
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import stats
import json
from raster_utils import load_and_match_raster, coord_to_index
from cost_surface import CostSurface
from path_analysis import PathAnalyzer
from pso_optimizer import PSOOptimizer


class AdvancedPreprocessor:
    
    def __init__(self):
        self.data_stats = {}
        self.transformation_params = {}
    
    def transform_slope(self, slope_raw, method='sqrt'):
        """
        Transformação para declividade: esticar valores baixos.
        
        Args:
            slope_raw: array bruto de declividade (graus)
            method: 'sqrt' (raiz quadrada) ou 'log' (logarítmica)
        
        Returns:
            array transformado
        """
        finite_mask = np.isfinite(slope_raw)
        finite_data = slope_raw[finite_mask]
        
        if len(finite_data) == 0:
            return slope_raw
        
        # Evitar valores negativos ou zero para transformações
        min_positive = np.min(finite_data[finite_data > 0])
        slope_safe = np.where(slope_raw <= 0, min_positive, slope_raw)
        
        if method == 'sqrt':
            # Transformação raiz quadrada: estica valores baixos
            transformed = np.sqrt(slope_safe)
            self.transformation_params['slope'] = {'method': 'sqrt', 'min_positive': float(min_positive)}
            
        elif method == 'log':
            # Transformação logarítmica: estica ainda mais valores baixos
            transformed = np.log1p(slope_safe)  # log1p(x) = log(1 + x)
            self.transformation_params['slope'] = {'method': 'log', 'min_positive': float(min_positive)}
        
        else:
            raise ValueError(f"Método de transformação não reconhecido: {method}")
        
        return transformed
    
    def analyze_insolation_distribution(self, insol_raw):
        """
        Analisa distribuição de insolação para detectar problemas.
        
        Args:
            insol_raw: array bruto de insolação
        
        Returns:
            dict com estatísticas da distribuição
        """
        finite_mask = np.isfinite(insol_raw)
        finite_data = insol_raw[finite_mask]
        
        stats_dict = {
            'total_pixels': len(finite_data),
            'zero_pixels': np.sum(finite_data == 0),
            'zero_percentage': 100 * np.sum(finite_data == 0) / len(finite_data),
            'min_value': float(np.min(finite_data)),
            'max_value': float(np.max(finite_data)),
            'mean_value': float(np.mean(finite_data)),
            'std_value': float(np.std(finite_data)),
            'unique_values': len(np.unique(finite_data))
        }
        
        # Detectar se é distribuição binária (muitos zeros)
        if stats_dict['zero_percentage'] > 50:
            stats_dict['distribution_type'] = 'binary_heavy'
            stats_dict['recommendation'] = 'categorization'
        elif stats_dict['unique_values'] < 10:
            stats_dict['distribution_type'] = 'categorical'
            stats_dict['recommendation'] = 'keep_categorical'
        else:
            stats_dict['distribution_type'] = 'continuous'
            stats_dict['recommendation'] = 'normalize_continuous'
        
        return stats_dict
    
    def transform_insolation(self, insol_raw, method='auto'):
        """
        Transformação para insolação baseada na análise da distribuição
        
        Args:
            insol_raw: array bruto de insolação
            method: 'auto' (detecta automaticamente), 'categorize', 'threshold', 'normalize'
        
        Returns:
            array transformado
        """
        stats = self.analyze_insolation_distribution(insol_raw)
        
        if method == 'auto':
            method = stats['recommendation']
        
        if method == 'categorize' or stats['distribution_type'] == 'binary_heavy':
            # Categorização: Baixa, Média, Alta
            finite_mask = np.isfinite(insol_raw)
            finite_data = insol_raw[finite_mask]
            
            if len(finite_data) == 0:
                return insol_raw
            
            # Usar percentis para categorização
            p33, p67 = np.percentile(finite_data[finite_data > 0], [33, 67])
            
            categorized = np.zeros_like(insol_raw)
            categorized[insol_raw == 0] = 0  # Baixa (sem sol)
            categorized[(insol_raw > 0) & (insol_raw <= p33)] = 0.33  # Baixa
            categorized[(insol_raw > p33) & (insol_raw <= p67)] = 0.67  # Média
            categorized[insol_raw > p67] = 1.0  # Alta
            
            self.transformation_params['insolation'] = {
                'method': 'categorize',
                'thresholds': [0.0, float(p33), float(p67), 1.0],
                'categories': ['Baixa', 'Média', 'Alta']
            }
            
        elif method == 'threshold':
            # Aplicar limiar para remover zeros excessivos
            finite_mask = np.isfinite(insol_raw)
            finite_data = insol_raw[finite_mask]
            
            if len(finite_data) == 0:
                return insol_raw
            
            # Definir limiar como percentil 10 dos valores > 0
            threshold = np.percentile(finite_data[finite_data > 0], 10)
            transformed = np.where(insol_raw < threshold, threshold, insol_raw)
            
            self.transformation_params['insolation'] = {
                'method': 'threshold',
                'threshold': float(threshold)
            }
            
        else:  # normalize
            # Normalização contínua padrão
            transformed = insol_raw.copy()
            self.transformation_params['insolation'] = {'method': 'normalize'}
        
        return transformed
    
    def apply_full_pipeline(self, dem_path, slope_path, insol_path, visib_path=None):
        """
        Aplica pipeline completo de pré-processamento
        
        Args:
            paths: caminhos para os arquivos raster
        
        Returns:
            dict com todos os dados processados e estatísticas
        """
        print("=" * 80)
        print("PIPELINE AVANÇADO DE PRÉ-PROCESSAMENTO LCP-PSO")
        print("=" * 80)
        
        # 1. Carregar dados brutos
        print("\n1. CARREGANDO DADOS BRUTOS...")
        
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype(float)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            dem_shape = dem.shape
            dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
        
        slope_raw = load_and_match_raster(slope_path, dem_shape, transform, crs)
        insol_raw = load_and_match_raster(insol_path, dem_shape, transform, crs)
        
        if visib_path:
            visib_raw = load_and_match_raster(visib_path, dem_shape, transform, crs)
        else:
            visib_raw = np.ones_like(dem)  # Visibilidade neutra
        
        print(f"   Dados carregados: shape {dem_shape}")
        
        # 2. Análise exploratória
        print("\n2. ANÁLISE EXPLORATÓRIA DOS DADOS...")
        
        # Análise da declividade
        slope_stats = {
            'mean': float(np.mean(slope_raw[np.isfinite(slope_raw)])),
            'std': float(np.std(slope_raw[np.isfinite(slope_raw)])),
            'min': float(np.min(slope_raw[np.isfinite(slope_raw)])),
            'max': float(np.max(slope_raw[np.isfinite(slope_raw)])),
            'skewness': float(stats.skew(slope_raw[np.isfinite(slope_raw)].flatten()))
        }
        
        # Análise da insolação
        insol_stats = self.analyze_insolation_distribution(insol_raw)
        
        print(f"   Declividade: média={slope_stats['mean']:.2f}°, skew={slope_stats['skewness']:.3f}")
        print(f"   Insolação: zeros={insol_stats['zero_percentage']:.1f}%, tipo={insol_stats['distribution_type']}")
        
        # 3. Transformações
        print("\n3. APLICANDO TRANSFORMAÇÕES ESPECIALIZADAS...")
        
        # Transformação da declividade (raiz quadrada recomendada para skew positivo)
        if slope_stats['skewness'] > 0.5:
            print("   Declividade: detectado skew positivo -> aplicando transformação raiz quadrada")
            slope_transformed = self.transform_slope(slope_raw, method='sqrt')
        else:
            print("   Declividade: distribuição aceitável -> mantendo valores originais")
            slope_transformed = slope_raw.copy()
        
        # Transformação da insolação baseada na distribuição
        print(f"   Insolação: {insol_stats['distribution_type']} -> {insol_stats['recommendation']}")
        insol_transformed = self.transform_insolation(insol_raw, method='auto')
        
        # 4. RETORNAR DADOS TRANSFORMADOS (sem normalização Min-Max)
        print("\n4. PREPARANDO DADOS TRANSFORMADOS PARA RANK NORMALIZATION...")
        
        # Não aplicar normalização Min-Max - deixar para CostSurface fazer Rank Normalization
        # slope_norm = self.normalize_minmax(slope_transformed)
        # visib_norm = self.normalize_minmax(visib_raw)
        # insol_norm = self.normalize_minmax(insol_transformed)
        
        # Verificação final dos dados transformados
        print("\n5. VERIFICAÇÃO FINAL DOS DADOS TRANSFORMADOS:")
        for name, data in [('Declividade', slope_transformed), ('Visibilidade', visib_raw), ('Insolação', insol_transformed)]:
            finite_data = data[np.isfinite(data)]
            print(f"   {name}: min={np.min(finite_data):.3f}, max={np.max(finite_data):.3f}, média={np.mean(finite_data):.3f}")
        
        # Compilar resultados
        processed_data = {
            'dem': dem,
            'slope_norm': slope_transformed,  # Dados transformados, não normalizados
            'visib_norm': visib_raw,          # Dados brutos, não normalizados
            'insol_norm': insol_transformed,    # Dados transformados, não normalizados
            'transform': transform,
            'crs': crs,
            'bounds': bounds,
            'shape': dem_shape,
            'slope_stats': slope_stats,
            'insol_stats': insol_stats,
            'transformation_params': self.transformation_params,
            'normalization_stats': self.data_stats
        }
        
        print("\n" + "=" * 80)
        print("PIPELINE DE PRÉ-PROCESSAMENTO CONCLUÍDO!")
        print("=" * 80)
        
        return processed_data


class DiversityAwarePSO(PSOOptimizer):
    """
    PSO com monitoramento de diversidade do enxame
    Implementa auditoria de convergência prematura
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diversity_history = []
        self.convergence_early_warning = False
    
    def calculate_diversity(self, positions):
        """
        Calcula diversidade do enxame como distância média entre partículas
        
        Args:
            positions: array com posições das partículas
        
        Returns:
            float: medida de diversidade (0 = todas iguais, 1 = máxima diversidade)
        """
        n_particles = positions.shape[0]
        
        if n_particles <= 1:
            return 0.0
        
        # Calcular distâncias euclidianas entre todos os pares
        total_distance = 0.0
        count = 0
        
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                distance = np.linalg.norm(positions[i] - positions[j])
                total_distance += distance
                count += 1
        
        if count == 0:
            return 0.0
        
        # Normalizar para [0, 1] (distância máxima possível = sqrt(n_dimensions))
        max_possible_distance = np.sqrt(self.n_dimensions)
        avg_distance = total_distance / count
        
        return avg_distance / max_possible_distance
    
    def optimize(self, path_analyzer, cost_surface_generator, start_coords, goal_coords,
                 alpha=1.0, sum_penalty_weight=10.0, verbose=True):
        """
        Otimização PSO com monitoramento de diversidade
        """
        # Inicializar partículas
        positions = np.random.uniform(0.1, 0.9, (self.n_particles, self.n_dimensions))
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, self.n_dimensions))
        
        pbest_positions = positions.copy()
        pbest_scores = np.full(self.n_particles, np.inf)
        
        gbest_position = None
        gbest_score = np.inf
        gbest_metrics = None
        
        print("\nINICIANDO OTIMIZAÇÃO PSO COM AUDITORIA DE DIVERSIDADE")
        print("-" * 60)
        
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                weights = positions[i]
                
                score = self.objective_function(
                    weights, path_analyzer, cost_surface_generator,
                    start_coords, goal_coords, alpha, sum_penalty_weight
                )
                
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()
                
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = weights.copy()
                    
                    # Calcular métricas detalhadas
                    cost_surface = cost_surface_generator.generate_cost_surface(weights)
                    gbest_metrics = path_analyzer.calculate_path_metrics(
                        cost_surface, start_coords, goal_coords
                    )
            
            # Atualizar velocidades e posições
            r1 = np.random.rand(self.n_particles, self.n_dimensions)
            r2 = np.random.rand(self.n_particles, self.n_dimensions)
            
            cognitive = self.c1 * r1 * (pbest_positions - positions)
            social = self.c2 * r2 * (gbest_position - positions)
            
            velocities = self.w_inertia * velocities + cognitive + social
            positions += velocities
            positions = np.clip(positions, 0.05, 0.95)
            
            # Perturbação para evitar estagnação
            if iteration % 10 == 0:
                positions += np.random.normal(0, 0.01, positions.shape)
                positions = np.clip(positions, 0.05, 0.95)
            
            # Calcular e registrar diversidade
            diversity = self.calculate_diversity(positions)
            self.diversity_history.append(diversity)
            
            # Detectar convergência prematura
            if diversity < 0.1 and iteration > 10:
                if not self.convergence_early_warning:
                    print(f"   ATENCAO: Baixa diversidade detectada na iteracao {iteration+1}!")
                    print(f"      Diversidade: {diversity:.3f} (ideal: > 0.2)")
                    print(f"      Recomendacao: aumentar inercia w ou reduzir coeficientes cognitivos")
                    self.convergence_early_warning = True
            
            # Registrar convergência
            self.convergence_history.append(gbest_score)
            
            if verbose and iteration % 5 == 0:
                weights_str = ', '.join([f'{w:.3f}' for w in gbest_position])
                print(f"Iter {iteration+1:3d}: pesos=[{weights_str}], score={gbest_score:.6f}, diversidade={diversity:.3f}")
        
        # Relatório final de diversidade
        avg_diversity = np.mean(self.diversity_history)
        min_diversity = np.min(self.diversity_history)
        
        print(f"\nRELATÓRIO DE DIVERSIDADE DO ENXAME:")
        print(f"   Diversidade média: {avg_diversity:.3f}")
        print(f"   Diversidade mínima: {min_diversity:.3f}")
        
        if min_diversity < 0.1:
            print("   ATENCAO: Enxame convergiu prematuramente!")
            print("   Sugestao: aumentar w_inertia para > 0.8 na proxima execucao")
        elif avg_diversity > 0.3:
            print("   OK: Diversidade saudavel mantida durante otimizacao")
        else:
            print("   OK: Diversidade aceitavel, mas pode ser melhorada")
        
        return {
            'best_weights': gbest_position,
            'best_score': gbest_score,
            'best_metrics': gbest_metrics,
            'convergence_history': self.convergence_history,
            'diversity_history': self.diversity_history,
            'avg_diversity': avg_diversity,
            'min_diversity': min_diversity,
            'early_convergence': self.convergence_early_warning
        }


if __name__ == "__main__":
    pass