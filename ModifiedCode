import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from anndata import AnnData
import pandas as pd
import numpy as np
import seaborn as sns
import colorcet as cc
import scanpy as sc
import igraph as ig
import colorcet as cc
import leidenalg
import umap
import copy
import os

class AnalysisConfig:
    """
    Configuration class for spatial analysis parameters.
    Centralizes all configurable parameters and provides documentation.
    """
    def __init__(self):
        # Graph parameters
        self.n_neighbors = 15
        self.include_self = False
        self.connectivity_mode = 'connectivity'
        
        # Visualization parameters
        self.edge_color = 'red'
        self.edge_width = 1
        self.point_color = 'blue'
        self.point_size = 20
        self.figsize = (15, 5)
        self.zoom_factor = 0.01
        
        # Color map parameters
        self.default_colormap = 'glasbey_bw_minc_20'

def glasbey(n_colours):
    """ Return a list of n RGB tuples representing colours from the Glasbey colour map"""
    cm = cc.cm.glasbey_bw_minc_20
    return [cm(val)[:3] for val in np.linspace(0, 1, cm.N)][:n_colours]

def get_colours_from(colourmap_name, n):
    """
    Return a list of n colours from the specified Matplotlib colourmap.
    
    Parameters:
    -----------
    colourmap_name : str
        Name of matplotlib colormap
    n : int
        Number of colors to generate
        
    Returns:
    --------
    list
        List of RGB color tuples
    """
    cmap = plt.get_cmap(colourmap_name)
    colours = [cmap(i / n) for i in range(n)]
    
    return colours

# function to plot the intensity of each marker on the dimensional reduction plots
def map_scatter(x, y, c, **kwargs):
    """
    Create scatter plot showing marker intensity on dimensional reduction plots.
    
    Parameters:
    -----------
    x : str
        Name of x-dimension column
    y : str
        Name of y-dimension column
    c : str
        Name of intensity column
    **kwargs : dict
        Additional keyword arguments for scatter plot
    """
    <c> is the name of the marker of interest
    Pass in the color and dataframe as well.
    """
    df = kwargs.pop("data")
    marker = df['Marker'].iloc[0]  # Get the marker name for the current facet
    vmin = df[c].min()
    vmax = df[c].max()
    kwargs.pop("color", None)  # Remove 'color' from kwargs if it exists
    kwargs.pop("vmin", None)
    kwargs.pop("vmax", None)
    scatter = plt.scatter(df[x], df[y], c=df[c], vmin=vmin, vmax=vmax, **kwargs)
    plt.colorbar(scatter, ax=plt.gca(), label=f'{marker} Intensity')

def leiden_cluster(embedding, res, config=None):
    """
    Perform Leiden clustering on spatial data using k-nearest neighbors.
    
    Parameters:
    -----------
    embedding : tuple of np.ndarray
        Coordinates for clustering
    res : float
        Resolution parameter for Leiden algorithm
    config : AnalysisConfig, optional
        Configuration object containing clustering parameters
        
    Returns:
    --------
    np.ndarray
        Array of cluster labels
    """
    if config is None:
        config = AnalysisConfig()
        
    # Create k-nearest neighbors graph
    knn_graph = kneighbors_graph(
        embedding, 
        n_neighbors=config.n_neighbors,
        mode=config.connectivity_mode,
        include_self=config.include_self
    )
    
    sources, targets = knn_graph.nonzero()
    edges = list(zip(sources.tolist(), targets.tolist()))

    # Create igraph graph from knn graph
    g = ig.Graph(edges=edges)

    # Perform Leiden clustering
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res)

    #Extracts custer labels 
    return np.array(partition.membership)

def plot_zoomed_triangulation_centered(ax, points, filtered_edges, title, zoom_factor=0.01, config=None):
    """
    Plot a zoomed-in section of the Delaunay triangulation centered on the middle.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    points : np.ndarray
        Array of point coordinates (n, 2)
    filtered_edges : list
        List of edges to plot
    title : str
        Plot title
    config : AnalysisConfig, optional
        Configuration object containing visualization parameters
    """

    if config is None:
        config = AnalysisConfig()    
    # zoom in on center of the graph
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    width = (max_x - min_x) * zoom_factor
    height = (max_y - min_y) * zoom_factor

    x_min_zoom = center_x - width / 2
    x_max_zoom = center_x + width / 2
    y_min_zoom = center_y - height / 2
    y_max_zoom = center_y + height / 2

    # filter points within the zoomed bounding box
    zoomed_points = points[
        (points[:, 0] >= x_min_zoom) & (points[:, 0] <= x_max_zoom) &
        (points[:, 1] >= y_min_zoom) & (points[:, 1] <= y_max_zoom)
    ]
    
    # map original points to their indices
    point_index_map = {tuple(point): idx for idx, point in enumerate(points)}
    zoomed_point_indices = {point_index_map[tuple(point)] for point in zoomed_points}

    # filter edges within the zoomed area
    zoomed_edges = [
        edge for edge in filtered_edges
        if edge[0] in zoomed_point_indices and edge[1] in zoomed_point_indices
    ]

    # Create plot
    for edge in zoomed_edges:
        x_values = [points[edge[0]][0], points[edge[1]][0]]
        y_values = [points[edge[0]][1], points[edge[1]][1]]
        ax.plot(x_values, y_values, color=config.edge_color, lw=config.edge_width)
    
    ax.scatter(zoomed_points[:, 0], zoomed_points[:, 1], 
              c=config.point_color, s=config.point_size)

    # set plot limits to the zoomed area
    ax.set_xlim(x_min_zoom, x_max_zoom)
    ax.set_ylim(y_min_zoom, y_max_zoom)
    ax.set_title(title)


def plot_zoomed_triangulation(ax, points, filtered_edges, title, x_min_zoom, x_max_zoom, y_min_zoom, y_max_zoom, config=None):
    """
    Plot a zoomed-in section of the Delaunay triangulation with filtered edges (2D array).

    Parameters:
    ax: matplotlib axis object to plot on
    points: numpy array of shape (n, 2), representing the coordinates of points
    filtered_edges: list of edges to plot, where each edge is a tuple of two point indices
    title: title of the plot
    x_min_zoom: minimum x value
    WLOG for x_max_zoom, y_min_zoom, y_max_zoom
    """
    if config is None:
        config = AnalysisConfig()    
    
    # filter points within the zoomed bounding box
    zoomed_points = points[
        (points[:, 0] >= x_min_zoom) & (points[:, 0] <= x_max_zoom) &
        (points[:, 1] >= y_min_zoom) & (points[:, 1] <= y_max_zoom)
    ]
    
    # map original points to their indices
    point_index_map = {tuple(point): idx for idx, point in enumerate(points)}
    zoomed_point_indices = {point_index_map[tuple(point)] for point in zoomed_points}

    # filter edges within the zoomed area
    zoomed_edges = [
        edge for edge in filtered_edges
        if edge[0] in zoomed_point_indices and edge[1] in zoomed_point_indices
    ]

    # plot
    for edge in zoomed_edges:
        x_values = [points[edge[0]][0], points[edge[1]][0]]
        y_values = [points[edge[0]][1], points[edge[1]][1]]
        ax.plot(x_values, y_values, color=config.edge_color, lw=config.edge_width)
    
    ax.scatter(zoomed_points[:, 0], zoomed_points[:, 1], 
              c=config.point_color, s=config.point_size)

    # set plot limits to the zoomed area   
    ax.set_xlim(x_min_zoom, x_max_zoom)
    ax.set_ylim(y_min_zoom, y_max_zoom)
    ax.set_title(title)


def plot_filtered_triangulation(ax, points, filtered_edges, title):
    """ Plot the Delaunay triangulation with filtered edges (2D array) """
    for edge in filtered_edges:
        x_values = [points[edge[0]][0], points[edge[1]][0]]
        y_values = [points[edge[0]][1], points[edge[1]][1]]
        ax.plot(x_values, y_values, 'r-', lw=1)
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=20)
    ax.set_title(title)

def filter_edges(tri, distance_cutoff, phenotype_mask):
     """
    Filter edges based on distance cutoff and phenotype mask.
    
    Parameters:
    -----------
    tri : scipy.spatial.Delaunay
        Delaunay triangulation object
    distance_cutoff : float
        Maximum distance allowed between connected points
    phenotype_mask : np.ndarray
        Boolean mask indicating valid phenotypes
    config : AnalysisConfig, optional
        Configuration object
        
    Returns:
    --------
    list
        Filtered edges meeting distance and phenotype criteria
    """
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edges.add(tuple(sorted([simplex[i], simplex[(i + 1) % 3]])))
    
    points = tri.points
    filtered_edges = []

    # create a mask to determine valid vertices
    valid_vertices = set()
    for vertex_index in range(len(points)):
        if phenotype_mask[vertex_index]:
            valid_vertices.add(vertex_index)

    for edge in edges:
        if edge[0] in valid_vertices and edge[1] in valid_vertices:
            p1, p2 = points[edge[0]], points[edge[1]]
            distance = np.linalg.norm(p1 - p2)
            if distance <= distance_cutoff:
                filtered_edges.append(edge)
    
    return filtered_edges

def analyze_slide(df, distance_cutoffs, phenotypes_oi, boundaries, config=None):
      """
    Analyze spatial relationships in slide data using Delaunay triangulation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing spatial and phenotype data
    distance_cutoffs : list
        List of distance thresholds for filtering edges
    phenotypes_oi : list
        List of phenotypes of interest
    boundaries : dict
        Dictionary mapping slide IDs to boundary coordinates (x_min, x_max, y_min, y_max)
    config : AnalysisConfig, optional
        Configuration object containing analysis parameters
    """
      if config is None:
        config = AnalysisConfig()    
  
    grouped = df.groupby('Parent')
    
    for name, group in grouped:
        coords = group[['Centroid X µm', 'Centroid Y µm']].values
        phenotypes = group['Phenotype'].values

        phenotype_mask = np.array([phenotype in phenotypes_oi for phenotype in phenotypes])
        
        # Delaunay triangulation
        tri = Delaunay(coords)
        
        # Set up plottinh
        num_cutoffs = len(distance_cutoffs)
        fig, axs = plt.subplots(1, num_cutoffs, figsize=config.figsize)
        if num_cutoffs == 1:
            axs = [axs]

        #Plot for each distance cut off
        for i, cutoff in enumerate(distance_cutoffs):
            filtered_edges = filter_edges(tri, cutoff, phenotype_mask) # get filtered graph
            x_min, x_max, y_min, y_max = boundaries[name]
            plot_zoomed_triangulation(ax=axs[i], 
                                      points=coords, 
                                      filtered_edges=filtered_edges, 
                                      title=f'Cutoff: {cutoff} µm', 
                                      x_min_zoom=x_min,
                                      x_max_zoom=x_max,
                                      y_min_zoom=y_min,
                                      y_max_zoom=y_max)
        
        plt.suptitle(f'Slide ID: {name}')
        plt.show()

def analyze_slide_zoomed_out(df, distance_cutoffs, phenotypes_oi):
    """ Analyze each slide, perform triangulation, filter edges for each cutoff, and plot. """
    grouped = df.groupby('Parent')
    
    for name, group in grouped:
        coords = group[['Centroid X µm', 'Centroid Y µm']].values
        phenotypes = group['Phenotype'].values

        phenotype_mask = np.array([phenotype in phenotypes_oi for phenotype in phenotypes])
        
        # Delaunay triangulation
        tri = Delaunay(coords)
        
        # plot for each distance cutoff
        num_cutoffs = len(distance_cutoffs)
        fig, axs = plt.subplots(1, num_cutoffs, figsize=(15, 5))
        if num_cutoffs == 1:
            axs = [axs]
        
        for i, cutoff in enumerate(distance_cutoffs):
            filtered_edges = filter_edges(tri, cutoff, phenotype_mask) # get filtered graph
            plot_filtered_triangulation(axs[i], coords, filtered_edges, f'Cutoff: {cutoff} µm')
        
        plt.suptitle(f'Slide ID: {name}')
        plt.show()
