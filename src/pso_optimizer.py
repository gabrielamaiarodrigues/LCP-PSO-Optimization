"""
Otimizador PSO para calibração de pesos de superfície de custo
Particle Swarm Optimization para encontrar pesos ótimos em análise LCP
"""
import numpy as np
from path_analysis import PathAnalyzer
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


class PSOOptimizer:
    """
    Otimizador PSO para encontrar pesos ótimos da superfície de custo
    """
    
    def __init__(self, n_particles=15, n_iterations=20, w_inertia=0.5, 
                 c1=2.0, c2=2.0, seed=42, n_dimensions=3):
        """
        Inicializa otimizador PSO.
        
        Args:
            n_particles: número de partículas
            n_iterations: número de iterações
            w_inertia: coeficiente de inércia
            c1: coeficiente cognitivo
            c2: coeficiente social
            seed: semente aleatória
            n_dimensions: número de dimensões (pesos)
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_inertia = w_inertia
        self.c1 = c1
        self.c2 = c2
        self.n_dimensions = n_dimensions
        
        np.random.seed(seed)
        
        # Histórico de convergência
        self.convergence_history = []
        
        # Histórico para animação
        self.positions_history = []
        self.gbest_history = []
        self.scores_history = []
    
    def objective_function(self, weights, path_analyzer, cost_surface_generator, start_coords, goal_coords, 
                          alpha=1.0, sum_penalty_weight=10.0, use_grass_mode=False, dem=None, regularization_strength=0.0):
        """
        Função objetivo do PSO com suporte a modo GRASS.
        
        Args:
            weights: pesos atuais [w_slope, w_visib, w_insol]
            path_analyzer: instância de PathAnalyzer
            cost_surface_generator: instância de CostSurface
            start_coords, goal_coords: coordenadas de início e fim
            alpha: peso do custo
            sum_penalty_weight: peso da penalização
            use_grass_mode: se True, usa grafo anisotrópico
            dem: array 2D com elevações
            regularization_strength: força da regularização
        """
        # Normalização forçada - Hard Constraint
        weights = weights / np.sum(weights)
        
        if use_grass_mode and dem is not None:
            # Usar superfícies separadas e grafo anisotrópico
            separate_surfaces = cost_surface_generator.get_separate_surfaces()
            metrics = path_analyzer.calculate_path_metrics(
                separate_surfaces, start_coords, goal_coords,
                use_grass_mode=True, dem=dem, weights=weights
            )
        else:
            # Usar superfície combinada
            cost_surface = cost_surface_generator.generate_cost_surface(weights)
            metrics = path_analyzer.calculate_path_metrics(cost_surface, start_coords, goal_coords)
        
        if metrics['cost'] == np.inf:
            return np.inf
        
        # Função objetivo: custo do caminho + regularização
        objective = alpha * metrics['cost']

        if regularization_strength and regularization_strength > 0:
            # KL divergence para evitar colapso de pesos
            n = len(weights)
            w = np.clip(weights, 1e-12, 1.0)
            kl_to_uniform = float(np.sum(w * np.log(w * n)))
            objective = objective * (1.0 + regularization_strength * kl_to_uniform)
        
        return objective
    
    def _evaluate_particle(self, args):
        """
        Função auxiliar para avaliação paralela de partículas
        """
        weights, path_analyzer, cost_surface_generator, start_coords, goal_coords, alpha, sum_penalty_weight, use_grass_mode, dem, regularization_strength = args
        
        return self.objective_function(
            weights, path_analyzer, cost_surface_generator,
            start_coords, goal_coords, alpha, sum_penalty_weight,
            use_grass_mode=use_grass_mode, dem=dem, regularization_strength=regularization_strength
        )
    
    def optimize(self, path_analyzer, cost_surface_generator, start_coords, goal_coords,
                 alpha=1.0, sum_penalty_weight=10.0, verbose=True, use_grass_mode=False, dem=None, parallel=True, regularization_strength=0.0):
        """
        Executa otimização PSO
        
        Args:
            path_analyzer: instância de PathAnalyzer
            cost_surface_generator: instância de CostSurface
            start_coords, goal_coords: coordenadas de início e fim
            alpha: peso do custo (fixado em 1.0)
            sum_penalty_weight: peso da penalização para soma ≠ 1
            verbose: se True, exibe progresso
            use_grass_mode: se True, usa grafo anisotrópico de 16 vizinhos
            dem: array 2D com elevações (necessário para modo GRASS)
        
        Returns:
            dict com melhores parâmetros e resultados
        """
        # Inicializar partículas com estratégia diversificada para evitar estagnação
        print(f"   [PSO] Inicializando {self.n_particles} partículas com diversificação...")
        
        # Estratégia 1: Partículas equilibradas (33% cada)
        n_balanced = self.n_particles // 3
        balanced_positions = np.random.dirichlet(np.ones(self.n_dimensions), n_balanced)
        
        # Estratégia 2: Partículas com foco em declividade (20%)
        n_slope_focus = max(1, self.n_particles // 5)
        slope_focus = np.random.dirichlet([3.0, 1.0, 1.0], n_slope_focus)  # Mais peso em slope
        
        # Estratégia 3: Partículas com foco em visibilidade (20%)
        n_visib_focus = max(1, self.n_particles // 5)
        visib_focus = np.random.dirichlet([1.0, 3.0, 1.0], n_visib_focus)  # Mais peso em visibilidade
        
        # Estratégia 4: Partículas aleatórias (restante)
        n_random = self.n_particles - n_balanced - n_slope_focus - n_visib_focus
        if n_random > 0:
            random_positions = np.random.dirichlet(np.ones(self.n_dimensions), n_random)
        else:
            random_positions = np.empty((0, self.n_dimensions))
        
        # Combinar todas as estratégias
        positions = np.vstack([balanced_positions, slope_focus, visib_focus, random_positions])
        
        # Velocidades iniciais pequenas para exploração gradual
        velocities = np.random.uniform(-0.05, 0.05, (self.n_particles, self.n_dimensions))
        
        print(f"   [PSO] Partículas inicializadas: {n_balanced} equilibradas, {n_slope_focus} foco-declividade, {n_visib_focus} foco-visibilidade, {n_random} aleatórias")
        
        # Melhores posições pessoais
        pbest_positions = positions.copy()
        pbest_scores = np.full(self.n_particles, np.inf)
        
        # Melhor posição global
        gbest_position = None
        gbest_score = np.inf
        gbest_metrics = None
        
        # Executar iterações
        for iteration in range(self.n_iterations):
            if parallel and self.n_particles >= 4:
                # Avaliação paralela das partículas
                args_list = [
                    (positions[i], path_analyzer, cost_surface_generator,
                     start_coords, goal_coords, alpha, sum_penalty_weight,
                     use_grass_mode, dem, regularization_strength)
                    for i in range(self.n_particles)
                ]
                
                with Pool(min(cpu_count(), self.n_particles)) as pool:
                    scores = pool.map(self._evaluate_particle, args_list)
            else:
                # Avaliação sequencial (fallback)
                scores = []
                for i in range(self.n_particles):
                    weights = positions[i]
                    score = self.objective_function(
                        weights, path_analyzer, cost_surface_generator,
                        start_coords, goal_coords, alpha, sum_penalty_weight,
                        use_grass_mode=use_grass_mode, dem=dem, regularization_strength=regularization_strength
                    )
                    scores.append(score)
            
            # Atualizar melhores posições pessoais e global
            for i in range(self.n_particles):
                score = scores[i]
                
                # Atualizar melhor pessoal
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()
                
                # Atualizar melhor global
                if score < gbest_score:
                    gbest_score = score
                    normalized_weights = positions[i] / np.sum(positions[i])
                    gbest_position = normalized_weights.copy()
                    
                    # Calcular métricas detalhadas para o melhor
                    if use_grass_mode and dem is not None:
                        separate_surfaces = cost_surface_generator.get_separate_surfaces()
                        gbest_metrics = path_analyzer.calculate_path_metrics(
                            separate_surfaces, start_coords, goal_coords,
                            use_grass_mode=True, dem=dem, weights=normalized_weights
                        )
                    else:
                        cost_surface = cost_surface_generator.generate_cost_surface(normalized_weights)
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
            
            # Manter partículas nos limites [0.05, 0.95] para evitar colapso
            positions = np.clip(positions, 0.05, 0.95)
            
            # MECANISMO ANTI-ESTAGNAÇÃO: Reinicialização parcial se não houver melhoria
            if iteration > 10 and iteration % 10 == 0:
                # Verificar se o melhor global melhorou nas últimas 10 iterações
                recent_improvement = gbest_score < (pbest_scores.mean() * 0.95)  # Pelo menos 5% de melhoria
                
                if not recent_improvement:
                    # Reiniciar 30% das piores partículas com estratégias diversificadas
                    n_reinit = max(3, self.n_particles // 3)
                    worst_indices = np.argsort(pbest_scores)[-n_reinit:]
                    
                    for idx in worst_indices:
                        # Reiniciar com estratégia aleatória diversificada
                        strategy = np.random.choice(['balanced', 'slope_focus', 'visib_focus'])
                        if strategy == 'balanced':
                            positions[idx] = np.random.dirichlet(np.ones(self.n_dimensions))
                        elif strategy == 'slope_focus':
                            positions[idx] = np.random.dirichlet([3.0, 1.0, 1.0])
                        else:  # visib_focus
                            positions[idx] = np.random.dirichlet([1.0, 3.0, 1.0])
                        
                        velocities[idx] = np.random.uniform(-0.05, 0.05, self.n_dimensions)
                        pbest_scores[idx] = np.inf  # Resetar melhor pessoal
                    
                    print(f"   [PSO] Reinicialização parcial na iteração {iteration}: {n_reinit} partículas reiniciadas")
            
            # Adicionar pequena perturbação ocasional para evitar estagnação
            if iteration % 10 == 0:
                positions += np.random.normal(0, 0.01, positions.shape)
                positions = np.clip(positions, 0.05, 0.95)
            
            # Registrar convergência e histórico para animação
            self.convergence_history.append(gbest_score)
            self.positions_history.append(positions.copy().flatten())
            self.gbest_history.append(gbest_position.copy())
            self.scores_history.append(gbest_score)
            
            if verbose and iteration % 5 == 0:
                weights_str = ', '.join([f'{w:.3f}' for w in gbest_position])
                mode_str = "GRASS" if use_grass_mode else "Original"
                print(f"Iteração {iteration+1}/{self.n_iterations} ({mode_str}): "
                      f"Melhor pesos=[{weights_str}], Score={gbest_score:.6f}")
        
        return {
            'best_weights': gbest_position,  # Array [w_slope, w_visib, w_insol]
            'best_score': gbest_score,
            'best_metrics': gbest_metrics,
            'convergence_history': self.convergence_history,
            'positions_history': self.positions_history,
            'gbest_history': self.gbest_history,
            'scores_history': self.scores_history
        }
