"""
Map Visualization Script for Next Location Prediction.

This script creates interactive map visualizations using Folium to demonstrate:
1. Input location sequences on a map
2. Prediction results (correct/incorrect)
3. Comparison between true and predicted locations

Based on visualization techniques from Helsinki City Bikes analysis.

Author: Research Team
Date: 2026-01-02
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import folium
from folium import plugins
import branca.colormap as cm


# =============================================================================
# Configuration
# =============================================================================

# Color palette (from Helsinki city bikes analysis)
COLORS = {
    'primary': '#8468F5',     # Purple
    'secondary': '#EC74E7',   # Pink
    'neutral': '#B2B2B2',     # Gray
    'warning': '#FFCB5C',     # Yellow
    'info': '#46B6E8',        # Blue
    'success': '#2FD4A1',     # Green
    'dark': '#333333',        # Dark
    'error': '#FF6B6B',       # Red
}

WEEKDAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


# =============================================================================
# Map Initialization Functions
# =============================================================================

def get_bounds(coords: List[Dict]) -> Tuple[float, float, float, float]:
    """Get bounding box from coordinates."""
    valid_coords = [c for c in coords if c.get('lat') is not None]
    if not valid_coords:
        return (0, 0, 0, 0)
    
    lats = [c['lat'] for c in valid_coords]
    lngs = [c['lng'] for c in valid_coords]
    
    return (min(lats), max(lats), min(lngs), max(lngs))


def init_map(center_lat: float, center_lng: float, zoom: int = 13) -> folium.Map:
    """Initialize a Folium map."""
    return folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom,
        tiles="cartodbpositron"
    )


def auto_zoom_map(m: folium.Map, coords: List[Dict]) -> folium.Map:
    """Auto-fit map to show all coordinates."""
    valid_coords = [(c['lat'], c['lng']) for c in coords if c.get('lat') is not None]
    if valid_coords:
        m.fit_bounds(valid_coords)
    return m


# =============================================================================
# Visualization Functions
# =============================================================================

def add_sequence_markers(m: folium.Map, sequence_coords: List[Dict], 
                         color: str = COLORS['primary'],
                         prefix: str = "Visit") -> folium.Map:
    """Add numbered markers for location sequence."""
    for i, coord in enumerate(sequence_coords):
        if coord.get('lat') is None:
            continue
        
        # Create numbered marker
        folium.CircleMarker(
            location=[coord['lat'], coord['lng']],
            radius=12,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{prefix} {i+1}</b><br>"
                f"Location ID: {coord.get('loc_id', 'N/A')}<br>"
                f"Lat: {coord['lat']:.6f}<br>"
                f"Lng: {coord['lng']:.6f}",
                max_width=200
            ),
            tooltip=f"{prefix} {i+1}"
        ).add_to(m)
        
        # Add number label
        folium.map.Marker(
            location=[coord['lat'], coord['lng']],
            icon=folium.DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=f'<div style="font-size: 10pt; color: white; font-weight: bold; text-align: center;">{i+1}</div>'
            )
        ).add_to(m)
    
    return m


def add_sequence_path(m: folium.Map, sequence_coords: List[Dict],
                      color: str = COLORS['primary'],
                      weight: int = 3,
                      dash_array: str = None) -> folium.Map:
    """Draw path connecting sequence locations."""
    valid_coords = [(c['lat'], c['lng']) for c in sequence_coords if c.get('lat') is not None]
    
    if len(valid_coords) >= 2:
        folium.PolyLine(
            locations=valid_coords,
            color=color,
            weight=weight,
            opacity=0.8,
            dash_array=dash_array
        ).add_to(m)
    
    return m


def add_target_marker(m: folium.Map, coords: Dict, 
                      is_correct: bool,
                      label: str = "Target") -> folium.Map:
    """Add marker for target location."""
    if coords.get('lat') is None:
        return m
    
    color = COLORS['success'] if is_correct else COLORS['error']
    icon_color = 'green' if is_correct else 'red'
    
    folium.Marker(
        location=[coords['lat'], coords['lng']],
        icon=folium.Icon(color=icon_color, icon='star', prefix='fa'),
        popup=folium.Popup(
            f"<b>{label}</b><br>"
            f"Location ID: {coords.get('loc_id', 'N/A')}<br>"
            f"Lat: {coords['lat']:.6f}<br>"
            f"Lng: {coords['lng']:.6f}",
            max_width=200
        ),
        tooltip=label
    ).add_to(m)
    
    return m


def add_prediction_marker(m: folium.Map, coords: Dict,
                         confidence: float,
                         is_correct: bool,
                         label: str = "Prediction") -> folium.Map:
    """Add marker for predicted location."""
    if coords.get('lat') is None:
        return m
    
    color = COLORS['success'] if is_correct else COLORS['warning']
    
    folium.CircleMarker(
        location=[coords['lat'], coords['lng']],
        radius=15,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        popup=folium.Popup(
            f"<b>{label}</b><br>"
            f"Location ID: {coords.get('loc_id', 'N/A')}<br>"
            f"Confidence: {confidence*100:.1f}%<br>"
            f"Correct: {'Yes' if is_correct else 'No'}<br>"
            f"Lat: {coords['lat']:.6f}<br>"
            f"Lng: {coords['lng']:.6f}",
            max_width=200
        ),
        tooltip=f"{label} ({confidence*100:.1f}%)"
    ).add_to(m)
    
    return m


def add_legend(m: folium.Map, is_correct: bool) -> folium.Map:
    """Add legend to map."""
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;">
        <b>Legend</b><br>
        <i class="fa fa-circle" style="color:{COLORS['primary']}"></i> Input Sequence<br>
        <i class="fa fa-star" style="color:{'green' if is_correct else 'red'}"></i> True Target<br>
        <i class="fa fa-circle" style="color:{COLORS['success'] if is_correct else COLORS['warning']}"></i> Model Prediction<br>
        <b>Result: {'✓ Correct' if is_correct else '✗ Incorrect'}</b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def add_info_panel(m: folium.Map, sample: Dict) -> folium.Map:
    """Add information panel to map."""
    weekday = WEEKDAY_NAMES[sample['weekdays'][-1] % 7] if sample.get('weekdays') else 'N/A'
    time_slot = sample['times'][-1] if sample.get('times') else 0
    hour = (time_slot * 15) // 60
    minute = (time_slot * 15) % 60
    time_str = f"{hour:02d}:{minute:02d}"
    
    info_html = f'''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 250px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;">
        <b>Sample Information</b><br>
        <hr style="margin: 5px 0;">
        User ID: {sample['user_id']}<br>
        Sequence Length: {len(sample['sequence'])}<br>
        Last Day: {weekday}<br>
        Last Time: {time_str}<br>
        <hr style="margin: 5px 0;">
        True Location: {sample['target_location']}<br>
        Predicted: {sample['predicted_location']}<br>
        Confidence: {sample['prediction_confidence']*100:.1f}%<br>
        Rank: {sample['rank']}<br>
        <hr style="margin: 5px 0;">
        <b style="color: {'green' if sample['is_correct'] else 'red'}">
        {'✓ CORRECT' if sample['is_correct'] else '✗ INCORRECT'}
        </b>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(info_html))
    return m


# =============================================================================
# Main Visualization Functions
# =============================================================================

def create_sample_visualization(sample: Dict, output_path: str = None) -> folium.Map:
    """Create full visualization for a single sample."""
    # Collect all coordinates for centering
    all_coords = sample['sequence_coords'].copy()
    if sample['target_coords'].get('lat') is not None:
        all_coords.append(sample['target_coords'])
    if sample['predicted_coords'].get('lat') is not None:
        all_coords.append(sample['predicted_coords'])
    
    # Calculate center
    valid_coords = [c for c in all_coords if c.get('lat') is not None]
    if not valid_coords:
        print(f"Warning: No valid coordinates for sample {sample['sample_idx']}")
        return None
    
    center_lat = np.mean([c['lat'] for c in valid_coords])
    center_lng = np.mean([c['lng'] for c in valid_coords])
    
    # Initialize map
    m = init_map(center_lat, center_lng, zoom=14)
    
    # Add sequence path
    m = add_sequence_path(m, sample['sequence_coords'], color=COLORS['primary'])
    
    # Add sequence markers
    m = add_sequence_markers(m, sample['sequence_coords'], color=COLORS['primary'])
    
    # Add target marker
    m = add_target_marker(m, sample['target_coords'], sample['is_correct'], label="True Target")
    
    # Add prediction marker (only if different from target)
    if not sample['is_correct']:
        m = add_prediction_marker(
            m, sample['predicted_coords'], 
            sample['prediction_confidence'],
            sample['is_correct'],
            label="Prediction"
        )
    
    # Add prediction line (from last sequence point to prediction)
    if sample['sequence_coords'] and sample['sequence_coords'][-1].get('lat') is not None:
        last_coord = sample['sequence_coords'][-1]
        
        # Line to prediction
        if sample['predicted_coords'].get('lat') is not None:
            folium.PolyLine(
                locations=[
                    [last_coord['lat'], last_coord['lng']],
                    [sample['predicted_coords']['lat'], sample['predicted_coords']['lng']]
                ],
                color=COLORS['warning'],
                weight=2,
                opacity=0.7,
                dash_array='10, 5'
            ).add_to(m)
    
    # Add legend and info panel
    m = add_legend(m, sample['is_correct'])
    m = add_info_panel(m, sample)
    
    # Auto-fit bounds
    m = auto_zoom_map(m, all_coords)
    
    # Save if path provided
    if output_path:
        m.save(output_path)
        print(f"Saved visualization to {output_path}")
    
    return m


def create_comparison_visualization(positive_sample: Dict, negative_sample: Dict,
                                   output_path: str = None) -> folium.Map:
    """Create side-by-side comparison visualization."""
    # This creates a single map showing both samples with different markers
    
    all_coords = []
    all_coords.extend(positive_sample['sequence_coords'])
    all_coords.extend(negative_sample['sequence_coords'])
    if positive_sample['target_coords'].get('lat'):
        all_coords.append(positive_sample['target_coords'])
    if negative_sample['target_coords'].get('lat'):
        all_coords.append(negative_sample['target_coords'])
    
    valid_coords = [c for c in all_coords if c.get('lat') is not None]
    if not valid_coords:
        return None
    
    center_lat = np.mean([c['lat'] for c in valid_coords])
    center_lng = np.mean([c['lng'] for c in valid_coords])
    
    m = init_map(center_lat, center_lng, zoom=12)
    
    # Add positive sample (green theme)
    m = add_sequence_path(m, positive_sample['sequence_coords'], color=COLORS['success'])
    m = add_target_marker(m, positive_sample['target_coords'], True, "Correct Prediction Target")
    
    # Add negative sample (red theme)
    m = add_sequence_path(m, negative_sample['sequence_coords'], color=COLORS['error'])
    m = add_target_marker(m, negative_sample['target_coords'], False, "Incorrect Prediction Target")
    
    m = auto_zoom_map(m, valid_coords)
    
    if output_path:
        m.save(output_path)
    
    return m


def visualize_all_demo_samples(demo_samples_path: str, output_dir: str):
    """Generate visualizations for all demo samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(demo_samples_path, 'r') as f:
        data = json.load(f)
    
    positive_samples = data.get('positive_samples', [])
    negative_samples = data.get('negative_samples', [])
    
    print(f"Generating visualizations...")
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Negative samples: {len(negative_samples)}")
    
    # Generate individual visualizations
    for i, sample in enumerate(positive_samples):
        output_path = os.path.join(output_dir, f'positive_sample_{i+1}.html')
        create_sample_visualization(sample, output_path)
    
    for i, sample in enumerate(negative_samples):
        output_path = os.path.join(output_dir, f'negative_sample_{i+1}.html')
        create_sample_visualization(sample, output_path)
    
    # Generate comparison visualization if we have both
    if positive_samples and negative_samples:
        comparison_path = os.path.join(output_dir, 'comparison.html')
        create_comparison_visualization(positive_samples[0], negative_samples[0], comparison_path)
    
    print(f"Visualizations saved to {output_dir}")


# =============================================================================
# Additional Visualization Functions
# =============================================================================

def create_heatmap_visualization(samples: List[Dict], output_path: str = None) -> folium.Map:
    """Create heatmap of all locations."""
    all_coords = []
    
    for sample in samples:
        for coord in sample['sequence_coords']:
            if coord.get('lat') is not None:
                all_coords.append([coord['lat'], coord['lng']])
        if sample['target_coords'].get('lat') is not None:
            all_coords.append([sample['target_coords']['lat'], sample['target_coords']['lng']])
    
    if not all_coords:
        return None
    
    center_lat = np.mean([c[0] for c in all_coords])
    center_lng = np.mean([c[1] for c in all_coords])
    
    m = init_map(center_lat, center_lng, zoom=11)
    
    plugins.HeatMap(all_coords, radius=15, blur=10).add_to(m)
    
    if output_path:
        m.save(output_path)
    
    return m


def create_cluster_visualization(samples: List[Dict], output_path: str = None) -> folium.Map:
    """Create marker cluster visualization."""
    all_coords = []
    
    for sample in samples:
        if sample['target_coords'].get('lat') is not None:
            all_coords.append({
                'lat': sample['target_coords']['lat'],
                'lng': sample['target_coords']['lng'],
                'is_correct': sample['is_correct'],
                'confidence': sample['prediction_confidence']
            })
    
    if not all_coords:
        return None
    
    center_lat = np.mean([c['lat'] for c in all_coords])
    center_lng = np.mean([c['lng'] for c in all_coords])
    
    m = init_map(center_lat, center_lng, zoom=11)
    
    marker_cluster = plugins.MarkerCluster().add_to(m)
    
    for coord in all_coords:
        color = 'green' if coord['is_correct'] else 'red'
        folium.Marker(
            location=[coord['lat'], coord['lng']],
            icon=folium.Icon(color=color, icon='info-sign'),
            popup=f"Correct: {coord['is_correct']}<br>Confidence: {coord['confidence']*100:.1f}%"
        ).add_to(marker_cluster)
    
    if output_path:
        m.save(output_path)
    
    return m


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Map visualization for next location prediction")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to demo_samples.json file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for visualizations")
    args = parser.parse_args()
    
    visualize_all_demo_samples(args.input, args.output_dir)


if __name__ == "__main__":
    main()
