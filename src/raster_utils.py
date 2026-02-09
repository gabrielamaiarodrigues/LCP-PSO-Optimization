"""
Utilitários para Processamento Raster
Funções para carregar, converter e manipular dados geoespaciais
"""
import numpy as np
import rasterio
from rasterio.transform import rowcol # converter coordenadas
from rasterio.warp import reproject, Resampling  # funções de reprojeção


def load_and_match_raster(raster_path, reference_shape, reference_transform, reference_crs):
    """
    Carrega raster e garante mesma forma/projeção que referência.
    
    Args:
        raster_path: caminho do arquivo raster
        reference_shape: shape (rows, cols) do raster de referência
        reference_transform: transform do raster de referência
        reference_crs: CRS do raster de referência
    
    Returns:
        numpy array com dados do raster reprojetado
    """
    with rasterio.open(raster_path) as src:
        dst = np.empty(reference_shape, dtype=np.float32)
        reproject(
            source=src.read(1).astype(float),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            resampling=Resampling.bilinear # método de interpolação bilinear (média dos 4 vizinhos mais próximos)
        )
        # Preencher NaNs com média
        dst = np.where(np.isnan(dst), np.nanmean(dst), dst)
        return dst


def normalize_robust(arr, min_val=1e-3, max_val=1.0):
    """
    Normalização robusta que evita problemas com valores extremos
    
    Args:
        arr: array numpy para normalizar
        min_val: valor mínimo após normalização
        max_val: valor máximo após normalização
    
    Returns:
        array normalizado entre [min_val, max_val]
    """
    mn, mx = np.nanpercentile(arr, [1, 99])  # Usa percentis para evitar outliers
    normalized = (arr - mn) / (mx - mn + 1e-12) # 1e-12: Evita divisão por zero
    return np.clip(normalized, min_val, max_val)


def coord_to_index(x, y, transform, shape):
    """
    Converte coordenadas geográficas para índices da matriz
    
    Args:
        x, y: coordenadas geográficas
        transform: transformação do raster
        shape: shape do raster (rows, cols)
    
    Returns:
        tuple (row, col) ou None se fora dos limites
    """
    try:
        row, col = rowcol(transform, x, y)
    except Exception:
        colf, rowf = ~transform * (x, y)
        row, col = int(rowf), int(colf)
    
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return None
    return (row, col)


def index_to_coord(row, col, transform):
    """
    Converte índices da matriz para coordenadas geográficas
    
    Args:
        row, col: índices da matriz
        transform: transformação do raster
    
    Returns:
        tuple (x, y)
    """
    x, y = transform * (col, row)
    return (x, y)