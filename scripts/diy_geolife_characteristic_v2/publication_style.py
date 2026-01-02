"""
Classic Scientific Publication Style Configuration.

This module provides consistent styling for all visualizations in the
DIY vs GeoLife characteristic analysis experiments.

Style Reference: Classic scientific publication style with:
- White background, black axis box (all 4 sides)
- Inside tick marks
- No grid lines
- Simple colors: black, blue, red, green
- Open markers: circles, squares, diamonds, triangles

Matches the reference images from:
- visualization_reference.png
- visualization_reference_2.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def setup_publication_style():
    """
    Configure matplotlib for classic scientific publication style.
    
    Call this at the start of each script to ensure consistent styling.
    """
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        
        # Figure settings
        'figure.figsize': (8, 6),
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        
        # Axes settings - box style (all 4 sides visible)
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.grid': False,  # No grid
        'axes.spines.top': True,  # Show all 4 sides
        'axes.spines.right': True,
        'axes.spines.bottom': True,
        'axes.spines.left': True,
        
        # Tick settings - inside ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.top': True,  # Ticks on all sides
        'xtick.bottom': True,
        'ytick.left': True,
        'ytick.right': True,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 7,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
    })


def setup_classic_axes(ax):
    """
    Configure axes to match classic scientific publication style.
    
    Args:
        ax: matplotlib axes object
    """
    # Ensure all spines visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # Inside ticks on all sides
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', which='major', length=5, width=1)
    ax.tick_params(axis='both', which='minor', length=3, width=0.5)


# Classic scientific color palette (matching reference)
COLORS = {
    'diy': 'blue',          # Blue for DIY dataset
    'geolife': 'red',       # Red for GeoLife dataset
    'black': 'black',
    'green': 'green',
    'orange': 'orange',
    'purple': '#9b59b6',
}

# Dataset-specific colors
DATASET_COLORS = {
    'DIY': 'blue',
    'GeoLife': 'red',
}

# Component colors for model analysis
COMPONENT_COLORS = {
    'pointer': 'green',     # Green for pointer mechanism
    'generation': '#9b59b6', # Purple for generation head
    'combined': 'orange',   # Orange for combined
}

# Marker styles (open markers like in reference)
MARKERS = {
    'diy': 'o',        # Circle
    'geolife': 's',    # Square
    'DIY': 'o',
    'GeoLife': 's',
}

# Hatch patterns for bar charts
HATCHES = {
    'diy': '///',
    'geolife': '\\\\\\',
    'DIY': '///',
    'GeoLife': '\\\\\\',
}


def get_dataset_style(dataset_name):
    """
    Get plotting style parameters for a dataset.
    
    Args:
        dataset_name: 'DIY' or 'GeoLife'
        
    Returns:
        dict with color, marker, hatch, linestyle
    """
    name = dataset_name.upper() if dataset_name.upper() in ['DIY', 'GEOLIFE'] else dataset_name
    name_key = 'DIY' if 'DIY' in name.upper() else 'GeoLife'
    
    return {
        'color': DATASET_COLORS.get(name_key, 'black'),
        'marker': MARKERS.get(name_key, 'o'),
        'hatch': HATCHES.get(name_key, ''),
        'linestyle': '-',
        'markerfacecolor': 'white',
        'markeredgewidth': 1.5,
        'linewidth': 1.5,
        'markersize': 7,
    }


def plot_line_with_markers(ax, x, y, label, dataset_name=None, color=None, marker=None, **kwargs):
    """
    Plot a line with open markers in classic scientific style.
    
    Args:
        ax: matplotlib axes
        x, y: data arrays
        label: legend label
        dataset_name: 'DIY' or 'GeoLife' (optional, for auto-styling)
        color, marker: override default style
        **kwargs: additional plot kwargs
    """
    if dataset_name:
        style = get_dataset_style(dataset_name)
        color = color or style['color']
        marker = marker or style['marker']
    
    color = color or 'black'
    marker = marker or 'o'
    
    ax.plot(x, y,
            marker=marker,
            color=color,
            label=label,
            linewidth=kwargs.get('linewidth', 1.5),
            markersize=kwargs.get('markersize', 7),
            markerfacecolor='white',
            markeredgecolor=color,
            markeredgewidth=kwargs.get('markeredgewidth', 1.5),
            linestyle=kwargs.get('linestyle', '-'))


def plot_bar_with_hatch(ax, x, heights, label, dataset_name=None, color=None, hatch=None, width=0.7, **kwargs):
    """
    Plot bars with hatching in classic scientific style.
    
    Args:
        ax: matplotlib axes
        x: bar positions
        heights: bar heights
        label: legend label
        dataset_name: 'DIY' or 'GeoLife' (optional, for auto-styling)
        color, hatch: override default style
        width: bar width
        **kwargs: additional bar kwargs
    """
    if dataset_name:
        style = get_dataset_style(dataset_name)
        color = color or style['color']
        hatch = hatch or style['hatch']
    
    color = color or 'black'
    hatch = hatch or ''
    
    bars = ax.bar(x, heights,
                  width=width,
                  label=label,
                  color='white',
                  edgecolor=color,
                  linewidth=1.5,
                  hatch=hatch,
                  **kwargs)
    
    return bars


def plot_scatter_open(ax, x, y, label, dataset_name=None, color=None, marker=None, **kwargs):
    """
    Plot scatter points with open markers in classic scientific style.
    
    Args:
        ax: matplotlib axes
        x, y: data arrays
        label: legend label
        dataset_name: 'DIY' or 'GeoLife' (optional, for auto-styling)
        color, marker: override default style
        **kwargs: additional scatter kwargs
    """
    if dataset_name:
        style = get_dataset_style(dataset_name)
        color = color or style['color']
        marker = marker or style['marker']
    
    color = color or 'black'
    marker = marker or 'o'
    
    ax.scatter(x, y,
               marker=marker,
               label=label,
               facecolors='white',
               edgecolors=color,
               linewidths=kwargs.get('linewidths', 1.5),
               s=kwargs.get('s', 60),
               **{k: v for k, v in kwargs.items() if k not in ['linewidths', 's']})


def create_legend(ax, loc='best', **kwargs):
    """Create a legend with classic scientific style."""
    ax.legend(loc=loc,
              frameon=True,
              edgecolor='black',
              fancybox=False,
              **kwargs)


def save_figure(fig, output_path, formats=['pdf', 'png']):
    """
    Save figure in multiple formats.
    
    Args:
        fig: matplotlib figure
        output_path: base path without extension
        formats: list of formats to save
    """
    from pathlib import Path
    output_path = Path(output_path)
    
    for fmt in formats:
        fig.savefig(
            output_path.with_suffix(f'.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """
    Add panel label (a), (b), etc. to subplot.
    
    Args:
        ax: matplotlib axes
        label: panel label string, e.g., '(a)'
        x, y: label position in axes coordinates
    """
    ax.text(x, y, label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            va='bottom',
            ha='left')
