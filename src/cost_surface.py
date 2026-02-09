"""
Geração de Superfície de Custo para LCP-PSO
Combina declividade, visibilidade e insolação em superfície de custo
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

class CostSurface:
    """
    Gera superfície de custo combinando múltiplas variáveis.

    - slope_grid: Maior valor = mais íngreme = MAIS CUSTO
    - insolation_grid: Maior valor = mais sol = MAIS CUSTO
    - visibility_grid: Maior valor = mais visível = MAIS CUSTO
    """

    def __init__(self, slope_grid, visibility_grid, insolation_grid, 
                 normalization_method='rank_uniform', base_friction=0.05):
        """
        Inicializa superfície de custo com normalização.
        
        Args:
            slope_grid: Declividade em graus [0, 90]
            visibility_grid: Índice de visibilidade [0, 1] onde 1 = muito visível
            insolation_grid: Insolação Wh/m² 
            base_friction: Custo mínimo [0.0, 1.0]
        """
        print(f"   [CostSurface] Inicializando com Rank Normalization (Base Friction={base_friction})...")
        
        self.critical_slope_threshold = 30.0
        self.impassable_mask = slope_grid > self.critical_slope_threshold
        self.base_friction = base_friction
        self.normalization_method = normalization_method
        
        print(f"     - Hard Cutoff: Declividade > {self.critical_slope_threshold}°")
        
        # Normalizar todas as variáveis como CUSTO (maior = pior)
        
        # Declividade: Maior = pior = mais custo
        self.slope_norm = self._rank_normalize(slope_grid, invert=False, mask=self.impassable_mask)
        
        # Insolação: Maior = pior = mais custo
        self.insol_norm = self._rank_normalize(insolation_grid, invert=False, mask=self.impassable_mask)
        
        # Visibilidade: Maior = mais visível = mais exposto = mais custo
        self.visib_norm = self._rank_normalize(visibility_grid, invert=False, mask=self.impassable_mask)
        
        # Aplicar infinitos para áreas intransitáveis
        self.slope_norm[self.impassable_mask] = np.inf
        self.visib_norm[self.impassable_mask] = np.inf
        self.insol_norm[self.impassable_mask] = np.inf
        
        # Diagnóstico
        valid_mask = ~self.impassable_mask
        print(f"     - Slope norm: mean={np.mean(self.slope_norm[valid_mask]):.4f}, "
              f"std={np.std(self.slope_norm[valid_mask]):.4f}")
        print(f"     - Visib norm: mean={np.mean(self.visib_norm[valid_mask]):.4f}, "
              f"std={np.std(self.visib_norm[valid_mask]):.4f}")
        print(f"     - Insol norm: mean={np.mean(self.insol_norm[valid_mask]):.4f}, "
              f"std={np.std(self.insol_norm[valid_mask]):.4f}")
        
        self.shape = slope_grid.shape

    def _rank_normalize(self, data, invert=False, mask=None):
        """
        Normalização por postos (rank-based) com distribuição uniforme.
        
        Args:
            data: Array de entrada
            invert: Se True, inverte a ordem dos postos
            mask: Máscara de áreas inválidas
        """
        if mask is not None:
            valid_data = data[~mask]
        else:
            valid_data = data.flatten()
            
        # Calcular ranks (1 a N)
        ranks = rankdata(valid_data, method='average')
        
        # Normalizar para [0, 1]
        norm_01 = (ranks - 1) / (len(ranks) - 1)
        
        # Inverter se necessário
        if invert:
            norm_01 = 1.0 - norm_01
            
        # Reescalar usando a fricção base
        norm_scaled = self.base_friction + (1.0 - self.base_friction) * norm_01
            
        # Recolocar no grid original
        result = np.zeros_like(data, dtype=np.float32)
        if mask is not None:
            result[~mask] = norm_scaled
            result[mask] = np.inf 
        else:
            result = norm_scaled.reshape(data.shape)
            
        return result

    def generate_cost_surface(self, weights, noise_std=0.0, noise_smooth=1.0):
        """
        Gera superfície de custo combinada.
        
        IMPORTANTE: Todas as variáveis JÁ estão orientadas como CUSTO:
        - slope_norm: maior = mais custo (OK)
        - visib_norm: maior = mais custo (SEM inversão dupla!) (OK)
        - insol_norm: maior = mais custo (OK)
        """
        w_slope, w_visib, w_insol = weights
        
        # Combinação linear DIRETA (sem inversões adicionais)
        cost = (w_slope * self.slope_norm) + (w_visib * self.visib_norm) + (w_insol * self.insol_norm)
        
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, self.shape)
            if noise_smooth > 0:
                noise = gaussian_filter(noise, sigma=noise_smooth)
            cost = cost * (1 + noise)
        
        cost = np.maximum(cost, 1e-5)
        cost[self.impassable_mask] = np.inf
        return cost
    
    def get_separate_surfaces(self):
        return np.stack([self.slope_norm, self.visib_norm, self.insol_norm])
        
    def get_individual_surfaces_for_visualization(self):
        """Retorna superfícies individuais para visualização."""
        return {
            'slope': self.slope_norm.copy(),
            'visibilidade': self.visib_norm.copy(),
            'insolacao': self.insol_norm.copy()
        }