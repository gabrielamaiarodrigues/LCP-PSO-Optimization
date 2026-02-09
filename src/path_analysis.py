"""
Análise de Caminhos para LCP-PSO
Calcula métricas e otimiza rotas usando algoritmos de grafos
"""
import numpy as np
import math
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from raster_utils import coord_to_index, index_to_coord
from skimage.graph import route_through_array

class PathAnalyzer:
    """
    Analisador de caminhos otimizado com vetorização NumPy.
    Calcula milhões de arestas usando operações matriciais.
    """
    
    def __init__(self, transform, shape):
        self.transform = transform
        self.shape = shape
        self.rows, self.cols = shape
        
        # Calcular tamanho do pixel em metros
        if abs(transform[0]) < 1.0: # Coordenadas em graus
            lat_center = transform[5] + (self.rows * transform[4]) / 2
            meters_per_deg_lat = 111320.0
            meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat_center))
            self.pixel_size = (abs(transform[0]) * meters_per_deg_lon + abs(transform[4]) * meters_per_deg_lat) / 2
        else:
            self.pixel_size = abs(transform[0])
            
        print(f"   [PathAnalyzer] Pixel size: {self.pixel_size:.2f}m")
        
        # Definir vizinhança (16 vizinhos)
        self.moves = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), # King
            (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)  # Knight
        ]
        
        # Pré-construir vetores de índices (vetorização)
        self._precompute_graph_topology()

    def _precompute_graph_topology(self):
        """Cria os vetores numpy de origem/destino para todas as arestas"""
        print("   [PathAnalyzer] Vetorizando topologia do grafo...")
        start_time = time.time()
        
        # Criar grids de coordenadas
        rows_grid, cols_grid = np.indices((self.rows, self.cols))
        rows_flat = rows_grid.flatten()
        cols_flat = cols_grid.flatten()
        
        src_indices = []
        dst_indices = []
        dist_values = []
        dr_values = []
        dc_values = []
        
        # Para cada movimento, calcular índices de todos os pixels válidos
        for dr, dc in self.moves:
            # Distância física deste movimento
            dist = math.sqrt(dr**2 + dc**2) * self.pixel_size
            
            # Calcular coordenadas de destino
            target_r = rows_flat + dr
            target_c = cols_flat + dc
            
            # Máscara de validade (dentro do grid)
            valid_mask = (target_r >= 0) & (target_r < self.rows) & (target_c >= 0) & (target_c < self.cols)
            
            # Filtrar
            valid_src = np.flatnonzero(valid_mask)
            valid_dst_r = target_r[valid_mask]
            valid_dst_c = target_c[valid_mask]
            valid_dst = valid_dst_r * self.cols + valid_dst_c
            
            src_indices.append(valid_src)
            dst_indices.append(valid_dst)
            dist_values.append(np.full_like(valid_src, dist, dtype=np.float32))
            dr_values.append(np.full_like(valid_src, dr, dtype=np.int8))
            dc_values.append(np.full_like(valid_src, dc, dtype=np.int8))
            
        # Concatenar tudo em vetores únicos gigantes
        self.all_src = np.concatenate(src_indices)
        self.all_dst = np.concatenate(dst_indices)
        self.all_dist = np.concatenate(dist_values)
        self.all_dr = np.concatenate(dr_values)
        self.all_dc = np.concatenate(dc_values)
        
        # Calcular coordenadas 2D para lookups rápidos
        self.src_r, self.src_c = np.divmod(self.all_src, self.cols)
        self.dst_r, self.dst_c = np.divmod(self.all_dst, self.cols)
        
        elapsed = time.time() - start_time
        print(f"   [PathAnalyzer] Topologia pronta: {len(self.all_src)} arestas em {elapsed:.2f}s")

    def build_anisotropic_cost_graph(self, cost_surface, dem, weights):
        start_time = time.time()
        
        # Extrair superfícies (já normalizadas [0,1])
        s_surf = cost_surface[0]  # Declividade: maior = mais custo
        v_surf = cost_surface[1]  # Visibilidade: maior = mais custo (JÁ CORRETO!)
        i_surf = cost_surface[2]  # Insolação: maior = mais custo
        
        base_cell = (weights[0] * s_surf) + (weights[1] * v_surf) + (weights[2] * i_surf)
        
        base_src = base_cell[self.src_r, self.src_c]
        base_dst = base_cell[self.dst_r, self.dst_c]
        base_cost = 0.5 * (base_src + base_dst)

        # Interpolação para movimentos Knight
        abs_dr = np.abs(self.all_dr)
        abs_dc = np.abs(self.all_dc)

        mask_knight_v = (abs_dr == 2) & (abs_dc == 1)
        if np.any(mask_knight_v):
            step_r = np.sign(self.all_dr[mask_knight_v]).astype(np.int8)
            step_c = np.sign(self.all_dc[mask_knight_v]).astype(np.int8)

            i1_r = self.src_r[mask_knight_v] + step_r
            i1_c = self.src_c[mask_knight_v]
            i2_r = self.src_r[mask_knight_v] + step_r
            i2_c = self.src_c[mask_knight_v] + step_c

            b_i1 = base_cell[i1_r, i1_c]
            b_i2 = base_cell[i2_r, i2_c]

            base_cost[mask_knight_v] = (base_src[mask_knight_v] + b_i1 + b_i2 + base_dst[mask_knight_v]) / 4.0

        mask_knight_h = (abs_dr == 1) & (abs_dc == 2)
        if np.any(mask_knight_h):
            step_r = np.sign(self.all_dr[mask_knight_h]).astype(np.int8)
            step_c = np.sign(self.all_dc[mask_knight_h]).astype(np.int8)

            i1_r = self.src_r[mask_knight_h]
            i1_c = self.src_c[mask_knight_h] + step_c
            i2_r = self.src_r[mask_knight_h] + step_r
            i2_c = self.src_c[mask_knight_h] + step_c

            b_i1 = base_cell[i1_r, i1_c]
            b_i2 = base_cell[i2_r, i2_c]

            base_cost[mask_knight_h] = (base_src[mask_knight_h] + b_i1 + b_i2 + base_dst[mask_knight_h]) / 4.0
        
        # 3. Calcular Anisotropia (Subida/Descida)
        z_src = dem[self.src_r, self.src_c]
        z_dst = dem[self.dst_r, self.dst_c]
        delta_z = z_dst - z_src
        
        # Inclinação = delta_z / distancia
        slopes = delta_z / self.all_dist
        slope_deg = np.degrees(np.arctan(slopes))

        # Fator Anisotrópico (Vetorizado)
        aniso_factor = np.ones_like(slopes, dtype=np.float32)

        mask_uphill = slope_deg > 0.0
        if np.any(mask_uphill):
            uphill_norm = np.clip(slope_deg[mask_uphill] / 30.0, 0.0, 3.0)
            k_up = 4.0
            aniso_factor[mask_uphill] = 1.0 + (uphill_norm * k_up) ** 2

        mask_moderate_down = (slope_deg <= -5.0) & (slope_deg >= -15.0)
        if np.any(mask_moderate_down):
            t = (np.abs(slope_deg[mask_moderate_down]) - 5.0) / 10.0
            aniso_factor[mask_moderate_down] = 0.95 - 0.10 * t

        mask_steep_down = slope_deg < -15.0
        if np.any(mask_steep_down):
            down_norm = np.clip((np.abs(slope_deg[mask_steep_down]) - 15.0) / 15.0, 0.0, 3.0)
            k_down = 2.0
            aniso_factor[mask_steep_down] = 1.0 + (down_norm * k_down) ** 2

        aniso_factor = np.clip(aniso_factor, 0.7, None)
        
        # 4. Custo Final da Aresta = Base * Anisotropia * Distância
        base_cost = np.maximum(base_cost, 1e-5)
        final_costs = base_cost * aniso_factor * self.all_dist

        # Bloqueio de declividades extremas
        max_slope_deg = 30.0
        blocked = np.abs(slope_deg) > max_slope_deg
        if np.any(blocked):
            final_costs[blocked] = np.inf
        
        # Criar matriz esparsa
        n_nodes = self.rows * self.cols
        finite = np.isfinite(final_costs)
        graph = csr_matrix((final_costs[finite], (self.all_src[finite], self.all_dst[finite])), shape=(n_nodes, n_nodes))
        
        elapsed = time.time() - start_time
        print(f"   [PathAnalyzer] Grafo construído em {elapsed:.3f}s")
        
        return graph

    def calculate_path_metrics(self, cost_surface, start_coords, goal_coords, 
                             use_grass_mode=False, dem=None, weights=None):
        """Calcula métricas do caminho (interface compatível)"""
        start_time = time.time()
        
        start_idx = coord_to_index(start_coords[0], start_coords[1], self.transform, self.shape)
        goal_idx = coord_to_index(goal_coords[0], goal_coords[1], self.transform, self.shape)
        
        if not start_idx or not goal_idx:
            return {'cost': np.inf, 'path': []}
            
        start_node = start_idx[0] * self.cols + start_idx[1]
        goal_node = goal_idx[0] * self.cols + goal_idx[1]
        
        # Construir grafo
        graph = self.build_anisotropic_cost_graph(cost_surface, dem, weights)
        
        # Dijkstra
        dist_matrix, predecessors = dijkstra(graph, indices=start_node, return_predecessors=True)
        
        if dist_matrix[goal_node] == np.inf:
            return {'cost': np.inf, 'path': []}
            
        # Reconstruir caminho
        path = []
        curr = goal_node
        while curr != -9999:
            path.append(curr)
            if curr == start_node: break
            curr = predecessors[curr]
            
        path = path[::-1] # Inverter
        
        # Converter para (r,c)
        path_indices = [(i // self.cols, i % self.cols) for i in path]
        
        # Calcular métricas extras
        path_array = np.array(path_indices)
        if len(path_array) >= 2:
            deltas = np.diff(path_array, axis=0)
            step_dist = np.sqrt(np.sum(deltas**2, axis=1)) * self.pixel_size
            length_m = float(np.sum(step_dist))
        else:
            length_m = 0.0
        
        # Curvatura
        curvatura = self.calculate_curvature(path_indices)
        
        # Eficiência
        start_pos = np.array(start_coords)
        goal_pos = np.array(goal_coords)
        direct_dist = np.sqrt(np.sum((goal_pos - start_pos)**2)) * 111000
        efficiency = direct_dist / length_m if length_m > 0 else 0
        
        elapsed = time.time() - start_time
        print(f"   [PathAnalyzer] Métricas calculadas em {elapsed:.3f}s")
        
        return {
            'cost': dist_matrix[goal_node],
            'length': length_m / 1000.0,
            'curvature': curvatura,
            'efficiency': efficiency,
            'path': path_indices
        }

    def calculate_curvature(self, path_indices):
        """Calcula curvatura do caminho"""
        if len(path_indices) < 3:
            return 0.0
            
        path_array = np.array(path_indices)
        vectors = np.diff(path_array, axis=0)
        
        angles = []
        for i in range(len(vectors) - 1):
            v1 = vectors[i]
            v2 = vectors[i + 1]
            
            dot_product = np.dot(v1, v2)
            mag1 = np.linalg.norm(v1)
            mag2 = np.linalg.norm(v2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        if angles:
            mean_angle = np.mean(angles)
            curvature = mean_angle * 180 / np.pi
        else:
            curvature = 0.0
            
        return curvature

    def find_path(self, cost_surface, start_coords, goal_coords, fully_connected=True, 
                use_grass_mode=False, dem=None, weights=None):
        """Encontra caminho de menor custo"""
        if use_grass_mode and dem is not None and weights is not None:
            metrics = self.calculate_path_metrics(cost_surface, start_coords, goal_coords,
                                                use_grass_mode=True, dem=dem, weights=weights)
            if metrics['cost'] != np.inf:
                return {
                    'path': metrics['path'],
                    'cost': metrics['cost'],
                    'cumulative_costs': [metrics['cost']]
                }
        
        # Fallback para modo original
        start_idx = coord_to_index(start_coords[0], start_coords[1], self.transform, self.shape)
        goal_idx = coord_to_index(goal_coords[0], goal_coords[1], self.transform, self.shape)
        if not start_idx or not goal_idx:
            return {
                'path': [],
                'cost': np.inf,
                'cumulative_costs': [np.inf]
            }

        start_rc = tuple(start_idx)
        goal_rc = tuple(goal_idx)
        path, cost = route_through_array(
            cost_surface,
            start_rc,
            goal_rc,
            fully_connected=fully_connected
        )
        
        return {
            'path': [tuple(p) for p in path],
            'cost': cost,
            'cumulative_costs': [cost]
        }

    def find_multiple_paths(self, cost_surfaces, start_coords, goal_coords, fully_connected=True):
        """Encontra múltiplos caminhos"""
        paths = []
        for i, surface in enumerate(cost_surfaces):
            result = self.find_path(surface, start_coords, goal_coords, fully_connected)
            result['path_id'] = i
            paths.append(result)
        return paths