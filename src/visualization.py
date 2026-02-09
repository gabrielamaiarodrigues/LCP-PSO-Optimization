"""
Visualização de Caminhos e Superfícies para LCP-PSO
Gera gráficos e mapas para análise de resultados
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LightSource
from rasterio.plot import show


class PathVisualizer:
    """
    Visualizador de caminhos e superfícies de custo.
    Gera mapas e gráficos para análise de resultados LCP.
    """
    
    def __init__(self, dem, transform, bounds):
        """
        Inicializa com dados do terreno
        
        Args:
            dem: array 2D com dados do DEM
            transform: transformação do raster
            bounds: limites geográficos do raster
        """
        self.dem = dem
        self.transform = transform
        self.bounds = bounds
    
    def plot_individual_cost_surfaces(self, cost_surface_obj, weights, 
                                      save_path=None, title_suffix=""):
        """
        Plota 3 superfícies de custo individuais.
        
        Args:
            cost_surface_obj: Instância de CostSurface
            weights: Array [w_slope, w_visib, w_insol]
            save_path: Caminho para salvar figura (não usado)
            title_suffix: Sufixo adicional para o título
        """
        # Obter superfícies individuais
        individual_surfaces = cost_surface_obj.get_individual_surfaces_for_visualization()
        
        # Coordenadas para extent
        extent = (self.bounds.left, self.bounds.right, 
                 self.bounds.bottom, self.bounds.top)
        
        # 1. Declividade
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        finite_slope = np.where(np.isfinite(individual_surfaces['slope']), 
                               individual_surfaces['slope'], np.nan)
        im1 = ax1.imshow(finite_slope, cmap='viridis_r', extent=extent, origin='upper')
        ax1.set_title("Declividade")
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Custo Normalizado', rotation=270, labelpad=15)
        plt.savefig('cost_slope.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Visibilidade
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        finite_visib = np.where(np.isfinite(individual_surfaces['visibilidade']), 
                               individual_surfaces['visibilidade'], np.nan)
        im2 = ax2.imshow(finite_visib, cmap='viridis_r', extent=extent, origin='upper')
        ax2.set_title("Visibilidade")
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Custo Normalizado', rotation=270, labelpad=15)
        plt.savefig('cost_visibility.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Insolação
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        finite_insol = np.where(np.isfinite(individual_surfaces['insolacao']), 
                               individual_surfaces['insolacao'], np.nan)
        im3 = ax3.imshow(finite_insol, cmap='viridis_r', extent=extent, origin='upper')
        ax3.set_title("Insolação")
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('Custo Normalizado', rotation=270, labelpad=15)
        plt.savefig('cost_insolation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   3 superficies de custo salvas: cost_slope.png, cost_visibility.png, cost_insolation.png")
    
    def plot_total_cost_surface(self, cost_surface_obj, weights, path_name):
        """
        Gera a superfície de custo total em PNG colorida.
        
        Args:
            cost_surface_obj: Instância de CostSurface
            weights: Array [w_slope, w_visib, w_insol]
            path_name: nome do caminho
        """
        # Gerar superfície combinada
        combined_surface = cost_surface_obj.generate_cost_surface(weights)
        
        # Coordenadas para extent
        extent = (self.bounds.left, self.bounds.right, 
                 self.bounds.bottom, self.bounds.top)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plotar superfície de custo total
        finite_combined = np.where(np.isfinite(combined_surface), combined_surface, np.nan)
        im = ax.imshow(finite_combined, cmap='viridis_r', extent=extent, origin='upper')
        
        ax.set_title('Superfície de Custo Total')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Custo Total', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(f'results_final/cost_total_{path_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Superficie de custo total salva: results_final/cost_total_{path_name}.png")
    
    def plot_single_path(self, path_metrics, cost_surface, start_coords, goal_coords, title="Caminho Otimizado", save_path=None):
        """
        Plota um único caminho sobre hillshade
        """
        if not path_metrics['path']:
            print("Nenhum caminho para plotar")
            return
        
        # Converter índices para coordenadas
        path_indices = np.array(path_metrics['path'])
        rows, cols = path_indices[:, 0], path_indices[:, 1]
        xs, ys = zip(*[self.transform * (col, row) for row, col in zip(rows, cols)])
        
        # Criar figura
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Hillshade com caminho
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.shade(self.dem, cmap=plt.cm.gray, vert_exag=1.5, blend_mode="overlay")
        
        axes[0].imshow(hillshade, extent=(self.bounds.left, self.bounds.right, 
                                         self.bounds.bottom, self.bounds.top), origin="upper")
        axes[0].plot(xs, ys, color="red", linewidth=2.0, label="Caminho Otimizado")
        
        axes[0].scatter([start_coords[0]], [start_coords[1]], c="lime", s=120, 
                       edgecolor="black", linewidth=1.5, label="Início", zorder=5, marker='o')
        axes[0].scatter([goal_coords[0]], [goal_coords[1]], c="red", s=120, 
                       edgecolor="black", linewidth=1.5, label="Fim", zorder=5, marker='o')
        
        axes[0].set_title(f"{title}\nCusto: {path_metrics['cost']:.2e}, "
                        f"Curvatura: {path_metrics['curvature']:.2f}°")
        axes[0].legend()
        
        # Superfície de custo com caminho
        finite_cost = np.where(np.isfinite(cost_surface), cost_surface, np.nan)
        im = axes[1].imshow(finite_cost, extent=(self.bounds.left, self.bounds.right,
                                                self.bounds.bottom, self.bounds.top), 
                           cmap='YlOrRd', origin='upper')
        axes[1].plot(xs, ys, color="blue", linewidth=2.0, label="Caminho")
        
        axes[1].scatter([start_coords[0]], [start_coords[1]], c="lime", s=120, 
                       edgecolor="black", linewidth=1.5, label="Início", zorder=5, marker='o')
        axes[1].scatter([goal_coords[0]], [goal_coords[1]], c="red", s=120, 
                       edgecolor="black", linewidth=1.5, label="Fim", zorder=5, marker='o')
        
        axes[1].set_title("Superfície de Custo")
        plt.colorbar(im, ax=axes[1], label="Custo Normalizado")
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Gráfico salvo em: {save_path}")
        
        plt.show()

    def plot_neighbor_grid(self, moves, save_path=None):
        """Plota a grade de vizinhos (king + knight)"""
        offsets = np.array(moves)
        if offsets.size == 0:
            print("Nenhuma vizinhança para plotar")
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(0, 0, s=120, c="black", label="Centro", zorder=3)
        for dr, dc in offsets:
            ax.plot([0, dc], [0, dr], color="gray", linewidth=1.0, alpha=0.6)
        ax.scatter(offsets[:, 1], offsets[:, 0], s=80, c="tab:blue", label="Vizinhos", zorder=2)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Delta col")
        ax.set_ylabel("Delta row")
        ax.set_title("Grade de vizinhos (16 conectividade)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   Grade de vizinhos salva em: {save_path}")

        plt.show()

    def plot_convergence(self, convergence_history, save_path=None):
        """Plota histórico de convergência do PSO"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(convergence_history, linewidth=2, color="blue")
        ax.set_xlabel("Iteração")
        ax.set_ylabel("Melhor Valor da Função Objetivo")
        ax.set_title("Convergência do PSO")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   Gráfico de convergência salvo em: {save_path}")
        
        plt.show()
