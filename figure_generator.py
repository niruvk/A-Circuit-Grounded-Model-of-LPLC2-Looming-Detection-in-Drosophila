"""
LPLC2 Paper Figure Generator
============================

Generates publication-quality figures for the LPLC2 model paper.
Run this script after preprocessing and model evaluation.

Usage:
    python generate_paper_figures.py

Output:
    ./paper_figures/ - Directory containing all figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import os
from glob import glob
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Use publication-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directories (adjust these to your setup)
PREPROCESSED_DIR = './lplc2_preprocessed'
OUTPUT_DIR = './paper_figures'
INPUTS_DIR = './lplc2_inputs'

# Model parameters (should match your model)
T4_WEIGHT = 0.01
T5_WEIGHT = 1.0
RF_RADIUS_DEG = 30.0
CENTER_SIGMA_DEG = 6.0
MULTIPLICATION_POWER = 0.75
GLOBAL_GAIN = 300.0
OPPONENT_THRESHOLD = 0.0
TAU_RISE = 0.050
TAU_DECAY = 0.300
SIGMOID_HALF = 100.0
SIGMOID_MAX = 100.0
FPS = 60
DT = 1.0 / FPS
DEG_TO_PX = 4.0

# Stimulus categories
FIG2D_STIMULI = {
    'fig2d_dark_rv10': 'Dark r/v=10ms',
    'fig2d_dark_rv20': 'Dark r/v=20ms',
    'fig2d_dark_rv40': 'Dark r/v=40ms',
    'fig2d_dark_rv80': 'Dark r/v=80ms',
    'fig2d_bright_rv10': 'Bright r/v=10ms',
    'fig2d_bright_rv20': 'Bright r/v=20ms',
    'fig2d_bright_rv40': 'Bright r/v=40ms',
    'fig2d_bright_rv80': 'Bright r/v=80ms',
}

FIG2F_STIMULI = {
    'fig2f_a_looming': 'Looming',
    'fig2f_b_bar_rightward': 'Bar →',
    'fig2f_c_edge_rightward': 'Edge →',
    'fig2f_d_bar_downward': 'Bar ↓',
    'fig2f_e_fullfield_downward': 'Full-field ↓',
    'fig2f_f_grating_rightward': 'Grating →',
    'fig2f_g_grating_downward': 'Grating ↓',
}

# Color scheme
COLORS = {
    'looming': '#2E7D32',  # Green
    'dark': '#212121',     # Black
    'bright': '#FF8F00',   # Orange
    'translation': '#1565C0',  # Blue
    'grating': '#7B1FA2',  # Purple
    'other': '#757575',    # Gray
    'highlight': '#D32F2F',  # Red for emphasis
}

# =============================================================================
# MODEL FUNCTIONS (copied from your model for standalone operation)
# =============================================================================

def create_quadrant_weights(height: int, width: int) -> Dict[str, np.ndarray]:
    """Create spatial weight maps for the four quadrants."""
    cy, cx = height // 2, width // 2
    y_grid, x_grid = np.ogrid[:height, :width]

    dx_deg = (x_grid - cx) / DEG_TO_PX
    dy_deg = (y_grid - cy) / DEG_TO_PX
    dist_deg = np.sqrt(dx_deg**2 + dy_deg**2)

    rf_weight = np.exp(-dist_deg**2 / (2 * CENTER_SIGMA_DEG**2))
    rf_weight[dist_deg > RF_RADIUS_DEG] = 0

    right_mask = 1 / (1 + np.exp(-dx_deg / 3))
    left_mask = 1 / (1 + np.exp(dx_deg / 3))
    top_mask = 1 / (1 + np.exp(dy_deg / 3))
    bottom_mask = 1 / (1 + np.exp(-dy_deg / 3))

    weights = {
        'right': (rf_weight * right_mask).astype(np.float32),
        'left': (rf_weight * left_mask).astype(np.float32),
        'top': (rf_weight * top_mask).astype(np.float32),
        'bottom': (rf_weight * bottom_mask).astype(np.float32),
        'full': rf_weight.astype(np.float32)
    }

    for key in weights:
        if weights[key].sum() > 0:
            weights[key] = weights[key] / weights[key].sum()

    return weights


def compute_directional_motion(flow: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract motion in each cardinal direction."""
    vx = flow[..., 0]
    vy = flow[..., 1]
    return {
        'rightward': np.maximum(0, vx),
        'leftward': np.maximum(0, -vx),
        'downward': np.maximum(0, vy),
        'upward': np.maximum(0, -vy),
    }


def run_model(preprocessed: Dict) -> Dict:
    """Run the LPLC2 model on preprocessed data."""
    on_signal = preprocessed['on_signal']
    off_signal = preprocessed['off_signal']
    flow = preprocessed['flow']

    n_frames, height, width = on_signal.shape
    weights = create_quadrant_weights(height, width)
    motion = T4_WEIGHT * on_signal + T5_WEIGHT * off_signal
    dir_motion = compute_directional_motion(flow)

    raw_response = np.zeros(n_frames, dtype=np.float32)
    opponent_signals = {
        'right': np.zeros(n_frames),
        'left': np.zeros(n_frames),
        'top': np.zeros(n_frames),
        'bottom': np.zeros(n_frames),
    }

    for t in range(n_frames):
        motion_t = motion[t]

        right_outward = np.sum(dir_motion['rightward'][t] * motion_t * weights['right'])
        right_inward = np.sum(dir_motion['leftward'][t] * motion_t * weights['right'])
        R = right_outward - right_inward

        left_outward = np.sum(dir_motion['leftward'][t] * motion_t * weights['left'])
        left_inward = np.sum(dir_motion['rightward'][t] * motion_t * weights['left'])
        L = left_outward - left_inward

        top_outward = np.sum(dir_motion['upward'][t] * motion_t * weights['top'])
        top_inward = np.sum(dir_motion['downward'][t] * motion_t * weights['top'])
        T = top_outward - top_inward

        bottom_outward = np.sum(dir_motion['downward'][t] * motion_t * weights['bottom'])
        bottom_inward = np.sum(dir_motion['upward'][t] * motion_t * weights['bottom'])
        B = bottom_outward - bottom_inward

        opponent_signals['right'][t] = R
        opponent_signals['left'][t] = L
        opponent_signals['top'][t] = T
        opponent_signals['bottom'][t] = B

        R_rect = max(0, R - OPPONENT_THRESHOLD)
        L_rect = max(0, L - OPPONENT_THRESHOLD)
        T_rect = max(0, T - OPPONENT_THRESHOLD)
        B_rect = max(0, B - OPPONENT_THRESHOLD)

        product = R_rect * L_rect * T_rect * B_rect
        raw_response[t] = GLOBAL_GAIN * (product ** MULTIPLICATION_POWER)

    saturated = SIGMOID_MAX * raw_response / (raw_response + SIGMOID_HALF + 1e-10)

    calcium = np.zeros(n_frames, dtype=np.float32)
    for t in range(1, n_frames):
        alpha = DT / TAU_RISE if saturated[t] > calcium[t-1] else DT / TAU_DECAY
        calcium[t] = calcium[t-1] + alpha * (saturated[t] - calcium[t-1])

    time = np.arange(n_frames) * DT - 1.0

    return {
        'raw': raw_response,
        'saturated': saturated,
        'calcium': calcium,
        'time': time,
        'peak': calcium.max(),
        'opponent_signals': opponent_signals,
    }


def load_preprocessed(name: str) -> Optional[Dict]:
    """Load preprocessed stimulus data."""
    p = os.path.join(PREPROCESSED_DIR, name)
    if not os.path.exists(p):
        return None

    result = {}
    for k in ['on_signal', 'off_signal', 'motion', 'flow']:
        f = os.path.join(p, f'{k}.npy')
        if os.path.exists(f):
            result[k] = np.load(f)

    radial_dir = os.path.join(p, 'radial')
    if os.path.exists(radial_dir):
        result['radial'] = {
            'outward': np.load(os.path.join(radial_dir, 'outward.npy')),
            'inward': np.load(os.path.join(radial_dir, 'inward.npy'))
        }

    required = ['on_signal', 'off_signal', 'flow']
    if not all(k in result for k in required):
        return None

    return result


def get_all_stimuli() -> List[str]:
    """Get list of all preprocessed stimuli."""
    if not os.path.exists(PREPROCESSED_DIR):
        return []

    stimuli = []
    for item in sorted(os.listdir(PREPROCESSED_DIR)):
        item_path = os.path.join(PREPROCESSED_DIR, item)
        if os.path.isdir(item_path):
            required = ['on_signal.npy', 'off_signal.npy', 'flow.npy']
            if all(os.path.exists(os.path.join(item_path, f)) for f in required):
                stimuli.append(item)
    return stimuli


# =============================================================================
# FIGURE GENERATION FUNCTIONS
# =============================================================================

def figure_2d_dark_vs_bright(results: Dict[str, Dict], output_dir: str):
    """
    Generate Figure 2D: Dark vs Bright Looming Comparison

    Creates a publication-quality figure showing:
    - Top: Calcium traces for dark and bright looming at different r/v values
    - Bottom: Summary bar chart with bright/dark ratio
    """
    rv_values = [10, 20, 40, 80]

    # Check which stimuli are available
    dark_available = [rv for rv in rv_values if f'fig2d_dark_rv{rv}' in results]
    bright_available = [rv for rv in rv_values if f'fig2d_bright_rv{rv}' in results]

    if not dark_available:
        print("  No Figure 2D stimuli found, skipping...")
        return None

    # Get peak values
    dark_peaks = [results[f'fig2d_dark_rv{rv}']['peak'] for rv in dark_available]
    bright_peaks = [results[f'fig2d_bright_rv{rv}']['peak'] for rv in bright_available]

    # Calculate statistics
    dark_mean = np.mean(dark_peaks) if dark_peaks else 0
    bright_mean = np.mean(bright_peaks) if bright_peaks else 0
    ratio = (bright_mean / dark_mean * 100) if dark_mean > 0 else 0

    # Create figure with 2 rows
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.35)

    # === Top panel: Traces ===
    gs_top = gs[0].subgridspec(2, len(dark_available), hspace=0.25, wspace=0.15)

    # Determine y-axis scale
    all_peaks = dark_peaks + bright_peaks
    y_max = max(all_peaks) * 1.15 if all_peaks else 100

    for i, rv in enumerate(dark_available):
        # Dark (top row)
        ax = fig.add_subplot(gs_top[0, i])
        r = results[f'fig2d_dark_rv{rv}']
        ax.plot(r['time'], r['calcium'], color=COLORS['dark'], lw=1.5)
        ax.fill_between(r['time'], 0, r['calcium'], color=COLORS['dark'], alpha=0.2)
        ax.set_xlim([r['time'][0], r['time'][-1]])
        ax.set_ylim([0, y_max])

        if i == 0:
            ax.set_ylabel('Dark\nΔF/F (%)', fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels([])

        # Add peak annotation
        ax.text(0.95, 0.95, f'{r["peak"]:.0f}%', transform=ax.transAxes,
               ha='right', va='top', fontsize=9, fontweight='bold')

        # Bright (bottom row) if available
        if rv in bright_available:
            ax = fig.add_subplot(gs_top[1, i])
            r = results[f'fig2d_bright_rv{rv}']
            ax.plot(r['time'], r['calcium'], color=COLORS['bright'], lw=1.5)
            ax.fill_between(r['time'], 0, r['calcium'], color=COLORS['bright'], alpha=0.2)
            ax.set_xlim([r['time'][0], r['time'][-1]])
            ax.set_ylim([0, y_max])
            ax.set_xlabel('Time (s)', fontsize=9)

            if i == 0:
                ax.set_ylabel('Bright\nΔF/F (%)', fontsize=10)
            else:
                ax.set_yticklabels([])

            ax.text(0.95, 0.95, f'{r["peak"]:.0f}%', transform=ax.transAxes,
                   ha='right', va='top', fontsize=9, fontweight='bold')

    # === Bottom panel: Summary ===
    gs_bottom = gs[1].subgridspec(1, 2, wspace=0.4)

    # Left: Peak vs r/v
    ax = fig.add_subplot(gs_bottom[0, 0])
    if dark_available:
        ax.plot(dark_available, dark_peaks, 'o-', color=COLORS['dark'],
               lw=2, ms=8, label='Dark')
    if bright_available:
        ax.plot(bright_available, bright_peaks, 's--', color=COLORS['bright'],
               lw=2, ms=8, label='Bright')
    ax.set_xscale('log')
    ax.set_xlabel('r/v (ms)', fontsize=11)
    ax.set_ylabel('Peak ΔF/F (%)', fontsize=11)
    ax.set_xlim([8, 100])
    ax.set_ylim([0, y_max])
    ax.legend(loc='upper right', frameon=False)

    # Right: Bar comparison
    ax = fig.add_subplot(gs_bottom[0, 1])
    bars = ax.bar(['Dark', 'Bright'], [dark_mean, bright_mean],
                 color=[COLORS['dark'], COLORS['bright']],
                 edgecolor='black', width=0.5)

    for bar, val in zip(bars, [dark_mean, bright_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
               f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Mean Peak ΔF/F (%)', fontsize=11)
    ax.set_ylim([0, dark_mean * 1.3 if dark_mean > 0 else 100])

    plt.savefig(os.path.join(output_dir, 'dark_vs_bright_looming.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'dark_vs_bright_looming.png'), format='png')
    plt.close()

    print(f"  Created: dark_vs_bright_looming.pdf/png (Bright/Dark = {ratio:.1f}%)")

    return {'dark_peaks': dark_peaks, 'bright_peaks': bright_peaks, 'ratio': ratio}


def figure_2f_selectivity(results: Dict[str, Dict], output_dir: str):
    """
    Generate Figure 2F: Stimulus Selectivity

    Creates a publication-quality figure showing:
    - Top: Calcium traces for each stimulus type
    - Bottom: Bar chart comparison with selectivity ratio
    """
    # Check available stimuli
    available = [(s, l) for s, l in FIG2F_STIMULI.items() if s in results]
    if not available:
        print("  No Figure 2F stimuli found, skipping...")
        return None

    stimuli, labels = zip(*available)
    peaks = [results[s]['peak'] for s in stimuli]

    # Calculate selectivity
    loom_peak = results.get('fig2f_a_looming', {}).get('peak', 0)
    other_peaks = [results[s]['peak'] for s in stimuli if 'looming' not in s]
    max_other = max(other_peaks) if other_peaks else 1
    selectivity = loom_peak / max_other if max_other > 0.01 else float('inf')

    n = len(available)

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[1.2, 1], hspace=0.35)

    # === Top panel: Individual traces ===
    gs_top = gs[0].subgridspec(1, n, wspace=0.15)

    y_max = max(peaks) * 1.15 if peaks else 100

    for i, (stim, label) in enumerate(available):
        ax = fig.add_subplot(gs_top[0, i])
        r = results[stim]

        # Color coding
        if 'looming' in stim:
            color = COLORS['looming']
        elif 'grating' in stim:
            color = COLORS['grating']
        elif 'bar' in stim or 'edge' in stim:
            color = COLORS['translation']
        else:
            color = COLORS['other']

        ax.plot(r['time'], r['calcium'], color='black', lw=1.5)
        ax.fill_between(r['time'], 0, r['calcium'], color=color, alpha=0.3)
        ax.set_xlim([r['time'][0], r['time'][-1]])
        ax.set_ylim([0, y_max])
        ax.set_xlabel('Time (s)', fontsize=9)

        if i == 0:
            ax.set_ylabel('ΔF/F (%)', fontsize=10)
        else:
            ax.set_yticklabels([])

        ax.text(0.95, 0.95, f'{r["peak"]:.0f}%', transform=ax.transAxes,
               ha='right', va='top', fontsize=9, fontweight='bold')

    # === Bottom panel: Bar chart ===
    ax = fig.add_subplot(gs[1])

    colors = []
    for s in stimuli:
        if 'looming' in s:
            colors.append(COLORS['looming'])
        elif 'grating' in s:
            colors.append(COLORS['grating'])
        elif 'bar' in s or 'edge' in s:
            colors.append(COLORS['translation'])
        else:
            colors.append(COLORS['other'])

    bars = ax.bar(range(len(peaks)), peaks, color=colors, edgecolor='black', width=0.7)

    for bar, p in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, p + 1,
               f'{p:.0f}', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(peaks)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Peak ΔF/F (%)', fontsize=11)
    ax.set_ylim([0, max(peaks) * 1.25 if peaks else 100])

    # Add 30% threshold line
    if loom_peak > 0:
        ax.axhline(loom_peak * 0.3, color=COLORS['highlight'], ls='--',
                  alpha=0.7, lw=1.5, label='30% of looming')
        ax.legend(loc='upper right', frameon=False)

    plt.savefig(os.path.join(output_dir, 'stimulus_selectivity.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'stimulus_selectivity.png'), format='png')
    plt.close()

    print(f"  Created: stimulus_selectivity.pdf/png (Selectivity = {selectivity:.1f}×)")

    return {'peaks': peaks, 'selectivity': selectivity}


def figure_opponent_signals(results: Dict[str, Dict], output_dir: str):
    """
    Generate Figure: Opponent Signal Analysis

    Shows how the multiplicative model discriminates between looming and non-looming.
    """
    # Select representative stimuli
    stim_list = [
        ('fig2f_a_looming', 'Looming'),
        ('fig2f_b_bar_rightward', 'Bar →'),
        ('fig2f_f_grating_rightward', 'Grating →'),
    ]

    available = [(s, l) for s, l in stim_list if s in results]
    if len(available) < 2:
        print("  Insufficient stimuli for opponent signal figure, skipping...")
        return None

    n = len(available)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(-1, 1)

    for i, (stim, label) in enumerate(available):
        r = results[stim]
        opp = r['opponent_signals']
        time = r['time']

        # Row 0: Raw opponent signals
        ax = axes[0, i]
        ax.plot(time, opp['right'], 'r-', lw=1.5, label='Right', alpha=0.8)
        ax.plot(time, opp['left'], 'b-', lw=1.5, label='Left', alpha=0.8)
        ax.plot(time, opp['top'], 'g-', lw=1.5, label='Top', alpha=0.8)
        ax.plot(time, opp['bottom'], 'm-', lw=1.5, label='Bottom', alpha=0.8)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_ylabel('Opponent Signal' if i == 0 else '', fontsize=10)
        if i == 0:
            ax.legend(fontsize=8, loc='upper left', ncol=2)

        # Row 1: Rectified signals
        ax = axes[1, i]
        R_rect = np.maximum(0, opp['right'])
        L_rect = np.maximum(0, opp['left'])
        T_rect = np.maximum(0, opp['top'])
        B_rect = np.maximum(0, opp['bottom'])

        ax.fill_between(time, 0, R_rect, alpha=0.4, color='r', label='R+')
        ax.fill_between(time, 0, L_rect, alpha=0.4, color='b', label='L+')
        ax.fill_between(time, 0, T_rect, alpha=0.4, color='g', label='T+')
        ax.fill_between(time, 0, B_rect, alpha=0.4, color='m', label='B+')
        ax.set_ylabel('Rectified [x]+' if i == 0 else '', fontsize=10)
        if i == 0:
            ax.legend(fontsize=8, loc='upper left', ncol=2)

        # Row 2: Final calcium response
        ax = axes[2, i]
        ax.plot(time, r['calcium'], 'k-', lw=2)
        ax.fill_between(time, 0, r['calcium'], color=COLORS['looming'] if 'looming' in stim else COLORS['other'], alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('ΔF/F (%)' if i == 0 else '', fontsize=10)
        ax.text(0.95, 0.95, f'Peak: {r["peak"]:.0f}%', transform=ax.transAxes,
               ha='right', va='top', fontsize=10, fontweight='bold')

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'opponent_signals.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'opponent_signals.png'), format='png')
    plt.close()

    print("  Created: opponent_signals.pdf/png")


def figure_model_schematic(output_dir: str):
    """
    Generate Figure: Model Schematic

    Visual diagram of the multiplicative radial opponency model.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Visual input
    ax.add_patch(plt.Rectangle((0.5, 7.5), 2, 1.2, fill=True,
                               facecolor='lightgray', edgecolor='black', lw=2))
    ax.text(1.5, 8.1, 'Visual\nInput', ha='center', va='center', fontsize=10)

    # Arrow to T4/T5
    ax.annotate('', xy=(3.5, 8.1), xytext=(2.5, 8.1),
               arrowprops=dict(arrowstyle='->', lw=2))

    # T4/T5 layer
    ax.add_patch(plt.Rectangle((3.5, 7.3), 2.5, 1.6, fill=True,
                               facecolor='lightyellow', edgecolor='black', lw=2))
    ax.text(4.75, 8.5, 'T4 (ON) + T5 (OFF)', ha='center', fontsize=9, fontweight='bold')
    ax.text(4.75, 7.9, 'Directional Motion', ha='center', fontsize=9)
    ax.text(4.75, 7.5, '→ ← ↑ ↓', ha='center', fontsize=12)

    # Arrow to quadrants
    ax.annotate('', xy=(7, 8.1), xytext=(6, 8.1),
               arrowprops=dict(arrowstyle='->', lw=2))

    # Four quadrants
    quad_colors = {'R': 'salmon', 'L': 'lightblue', 'T': 'lightgreen', 'B': 'plum'}
    quad_positions = [(7.2, 7.8), (8.2, 7.8), (7.2, 6.8), (8.2, 6.8)]
    quad_labels = ['R+', 'L+', 'T+', 'B+']
    quad_formulas = ['→ - ←', '← - →', '↑ - ↓', '↓ - ↑']

    for pos, label, formula, (name, color) in zip(quad_positions, quad_labels, quad_formulas, quad_colors.items()):
        ax.add_patch(plt.Rectangle(pos, 0.9, 0.9, fill=True,
                                   facecolor=color, edgecolor='black', lw=1.5))
        ax.text(pos[0]+0.45, pos[1]+0.6, label, ha='center', fontsize=10, fontweight='bold')
        ax.text(pos[0]+0.45, pos[1]+0.25, formula, ha='center', fontsize=8)

    ax.text(7.7, 6.5, 'Opponent Signals', ha='center', fontsize=9, style='italic')

    # Arrow to multiplication
    ax.annotate('', xy=(7.7, 5.8), xytext=(7.7, 6.4),
               arrowprops=dict(arrowstyle='->', lw=2))

    # Multiplication
    ax.add_patch(plt.Rectangle((6.7, 4.8), 2, 0.9, fill=True,
                               facecolor='lightyellow', edgecolor='black', lw=2))
    ax.text(7.7, 5.4, '× × ×', ha='center', fontsize=14, fontweight='bold')
    ax.text(7.7, 5.0, 'Multiplicative', ha='center', fontsize=9)

    # Arrow to output
    ax.annotate('', xy=(7.7, 4.0), xytext=(7.7, 4.7),
               arrowprops=dict(arrowstyle='->', lw=2))

    # Output
    ax.add_patch(plt.Rectangle((6.4, 3.0), 2.6, 0.9, fill=True,
                               facecolor='lightgreen', edgecolor='black', lw=2))
    ax.text(7.7, 3.6, 'LPLC2 Response', ha='center', fontsize=10, fontweight='bold')
    ax.text(7.7, 3.2, 'G·(R+·L+·T+·B+)^p', ha='center', fontsize=9, family='monospace')

    # Equation box
    ax.add_patch(plt.Rectangle((0.5, 3.0), 5, 3.5, fill=True,
                               facecolor='white', edgecolor='black', lw=1.5))
    eq_text = (
        "Key Equations:\n\n"
        "Opponent signal:\n"
        "  $S_q = O_q^{out} - O_q^{in}$\n\n"
        "Rectification:\n"
        "  $S_q^+ = max(0, S_q)$\n\n"
        "Response:\n"
        "  $R = G \\cdot (R^+ \\cdot L^+ \\cdot T^+ \\cdot B^+)^p$"
    )
    ax.text(0.7, 6.3, eq_text, ha='left', va='top', fontsize=9, family='monospace')

    # Key insight
    ax.text(5, 1.5,
           'Key Insight: Multiplicative integration requires\n'
           'ALL quadrants positive → selective for radial expansion',
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_schematic.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'model_schematic.png'), format='png')
    plt.close()

    print("  Created: model_schematic.pdf/png")


def figure_bar_expansion(results: Dict[str, Dict], output_dir: str):
    """
    Generate Figure: Bar Expansion Analysis (Figure 3/4 from paper)
    """
    # Bar expansion stimuli
    bar_stimuli = [s for s in results.keys() if 'bar' in s.lower() and 'fig3' in s.lower()]

    if not bar_stimuli:
        print("  No bar expansion stimuli found, skipping...")
        return None

    # Group by width
    widths = [10, 60]
    orientations = [0, 45, 90, 135]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for row, width in enumerate(widths):
        for col, orient in enumerate(orientations):
            ax = axes[row, col]

            # Try to find matching stimulus
            key = f"fig3{'e' if width==10 else 'f'}_bar{width}_orient{orient}"

            if key in results:
                r = results[key]
                ax.plot(r['time'], r['calcium'], 'k-', lw=1.5)
                ax.fill_between(r['time'], 0, r['calcium'],
                               color=COLORS['translation'], alpha=0.3)
                ax.set_xlim([r['time'][0], r['time'][-1]])
                ax.text(0.95, 0.95, f'{r["peak"]:.0f}%', transform=ax.transAxes,
                       ha='right', va='top', fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)

            if col == 0:
                ax.set_ylabel(f'{width}° bar\nΔF/F (%)', fontsize=10)
            if row == 1:
                ax.set_xlabel('Time (s)', fontsize=9)

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'bar_expansion.pdf'), format='pdf')
    plt.savefig(os.path.join(output_dir, 'bar_expansion.png'), format='png')
    plt.close()

    print("  Created: bar_expansion.pdf/png")


def generate_results_table(results: Dict[str, Dict], output_dir: str):
    """
    Generate LaTeX table with all results.
    """
    # Collect all peaks
    all_data = []

    for name, r in sorted(results.items()):
        category = 'Other'
        if 'dark' in name.lower():
            category = 'Dark Looming'
        elif 'bright' in name.lower():
            category = 'Bright Looming'
        elif 'looming' in name.lower():
            category = 'Looming'
        elif 'grating' in name.lower():
            category = 'Grating'
        elif 'bar' in name.lower():
            category = 'Bar/Edge'
        elif 'fullfield' in name.lower() or 'translation' in name.lower():
            category = 'Translation'

        all_data.append({
            'name': name,
            'category': category,
            'peak': r['peak'],
        })

    # Generate LaTeX table
    latex = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Model Response Summary}",
        "\\label{tab:results}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Stimulus & Category & Peak ΔF/F (\\%) \\\\",
        "\\midrule",
    ]

    for d in all_data:
        latex.append(f"{d['name'].replace('_', '\\_')} & {d['category']} & {d['peak']:.1f} \\\\")

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    # Save with UTF-8 encoding to handle special characters
    with open(os.path.join(output_dir, 'results_table.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex))

    print("  Created: results_table.tex")

    # Also save as markdown
    md = [
        "| Stimulus | Category | Peak dF/F (%) |",
        "|----------|----------|---------------|",
    ]
    for d in all_data:
        md.append(f"| {d['name']} | {d['category']} | {d['peak']:.1f} |")

    with open(os.path.join(output_dir, 'results_table.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print("  Created: results_table.md")


def generate_summary_statistics(results: Dict[str, Dict], output_dir: str):
    """
    Generate summary statistics file.
    """
    stats = {
        'total_stimuli': len(results),
    }

    # Figure 2D stats
    dark_peaks = [results[f'fig2d_dark_rv{rv}']['peak']
                 for rv in [10, 20, 40, 80] if f'fig2d_dark_rv{rv}' in results]
    bright_peaks = [results[f'fig2d_bright_rv{rv}']['peak']
                   for rv in [10, 20, 40, 80] if f'fig2d_bright_rv{rv}' in results]

    if dark_peaks and bright_peaks:
        stats['dark_mean'] = np.mean(dark_peaks)
        stats['bright_mean'] = np.mean(bright_peaks)
        stats['bright_dark_ratio'] = stats['bright_mean'] / stats['dark_mean'] * 100

    # Figure 2F stats
    if 'fig2f_a_looming' in results:
        loom = results['fig2f_a_looming']['peak']
        others = [results[s]['peak'] for s in FIG2F_STIMULI if s in results and 'looming' not in s]
        if others:
            stats['looming_peak'] = loom
            stats['max_other'] = max(others)
            stats['selectivity'] = loom / max(others) if max(others) > 0.01 else float('inf')

    # Write to file
    lines = [
        "=" * 60,
        "LPLC2 MODEL RESULTS SUMMARY",
        "=" * 60,
        "",
        f"Total stimuli processed: {stats.get('total_stimuli', 0)}",
        "",
        "--- Figure 2D: Dark vs Bright ---",
        f"Dark mean peak:       {stats.get('dark_mean', 'N/A'):.1f}%" if 'dark_mean' in stats else "Dark mean peak:       N/A",
        f"Bright mean peak:     {stats.get('bright_mean', 'N/A'):.1f}%" if 'bright_mean' in stats else "Bright mean peak:     N/A",
        f"Bright/Dark ratio:    {stats.get('bright_dark_ratio', 'N/A'):.1f}%" if 'bright_dark_ratio' in stats else "Bright/Dark ratio:    N/A",
        f"Target range:         20-30%",
        f"Status:               {'PASS' if 'bright_dark_ratio' in stats and 20 <= stats['bright_dark_ratio'] <= 30 else 'CHECK'}",
        "",
        "--- Figure 2F: Selectivity ---",
        f"Looming peak:         {stats.get('looming_peak', 'N/A'):.1f}%" if 'looming_peak' in stats else "Looming peak:         N/A",
        f"Max other peak:       {stats.get('max_other', 'N/A'):.1f}%" if 'max_other' in stats else "Max other peak:       N/A",
        f"Selectivity ratio:    {stats.get('selectivity', 'N/A'):.1f}×" if 'selectivity' in stats and stats['selectivity'] != float('inf') else "Selectivity ratio:    ∞",
        f"Target:               >3×",
        f"Status:               {'PASS' if 'selectivity' in stats and stats['selectivity'] > 3 else 'CHECK'}",
        "",
        "--- Model Parameters ---",
        f"T4 weight:            {T4_WEIGHT}",
        f"T5 weight:            {T5_WEIGHT}",
        f"RF radius:            {RF_RADIUS_DEG}°",
        f"Center sigma:         {CENTER_SIGMA_DEG}°",
        f"Multiplication power: {MULTIPLICATION_POWER}",
        f"Global gain:          {GLOBAL_GAIN}",
        "",
        "=" * 60,
    ]

    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print("  Created: summary_statistics.txt")

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate all paper figures."""
    print("=" * 60)
    print("LPLC2 Paper Figure Generator")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Get all stimuli
    stimuli = get_all_stimuli()
    print(f"Found {len(stimuli)} preprocessed stimuli")

    if not stimuli:
        print("\nERROR: No preprocessed stimuli found!")
        print(f"Please ensure {PREPROCESSED_DIR} contains preprocessed data.")
        print("Run your preprocessing pipeline first.")
        return

    # Run model on all stimuli
    print("\nRunning model on all stimuli...")
    results = {}
    for name in stimuli:
        preprocessed = load_preprocessed(name)
        if preprocessed is not None:
            results[name] = run_model(preprocessed)
            print(f"  {name}: peak = {results[name]['peak']:.1f}%")

    print(f"\nProcessed {len(results)} stimuli successfully")

    # Generate figures
    print("\nGenerating figures...")

    # Figure 2D
    fig2d_stats = figure_2d_dark_vs_bright(results, OUTPUT_DIR)

    # Figure 2F
    fig2f_stats = figure_2f_selectivity(results, OUTPUT_DIR)

    # Opponent signals
    figure_opponent_signals(results, OUTPUT_DIR)

    # Model schematic
    figure_model_schematic(OUTPUT_DIR)

    # Bar expansion
    figure_bar_expansion(results, OUTPUT_DIR)

    # Tables and statistics
    generate_results_table(results, OUTPUT_DIR)
    stats = generate_summary_statistics(results, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if fig2d_stats:
        print(f"Dark vs Bright - Bright/Dark ratio: {fig2d_stats['ratio']:.1f}%", end=' ')
        print('✓' if 20 <= fig2d_stats['ratio'] <= 30 else '(target: 20-30%)')

    if fig2f_stats:
        sel = fig2f_stats['selectivity']
        if sel == float('inf'):
            print("Stimulus Selectivity: ∞ ✓")
        else:
            print(f"Stimulus Selectivity: {sel:.1f}×", end=' ')
            print('✓' if sel > 3 else '(target: >3×)')

    print(f"\nAll figures saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()