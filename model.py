"""
LPLC2 Model - Pure Multiplicative Radial Opponency
===================================================

Based on Klapoetke et al. 2017 (Nature) and Zhao et al. 2023 (iScience).

Key mechanism: MULTIPLICATIVE INTEGRATION across 4 directional channels.
This provides STRICT selectivity for looming - only stimuli with outward motion
in ALL 4 quadrants simultaneously will activate the detector.

LPLC2 dendrites span all 4 lobula plate layers, each encoding one cardinal direction.
For each spatial quadrant, we compute an "opponent signal":
  - Right quadrant: rightward_motion - leftward_motion
  - Left quadrant: leftward_motion - rightward_motion
  - Top quadrant: upward_motion - downward_motion
  - Bottom quadrant: downward_motion - upward_motion

Response = GLOBAL_GAIN × (R+ × L+ × T+ × B+)^POWER

Higher POWER (0.5) penalizes weak quadrants more than lower power (0.25):
- Looming: all 4 quadrants strong → (high × high × high × high)^0.5 = HIGH
- Bar expansion: 2 strong, 2 weak → (high × high × low × low)^0.5 = MODERATE
- Translation: 1 strong, 3 weak → product ≈ 0

CENTER_SIGMA controls spatial weighting:
- Smaller sigma favors narrow bars (10°) over wide bars (60°)
- Larger sigma lets looming benefit from full RF coverage

Parameters from literature:
- RF diameter: ~60°
- T4/T5 ratio: ~45%/55% (T5 dominant for dark preference)
- Temporal: τ_rise ≈ 50ms, τ_decay ≈ 300ms
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from glob import glob
from typing import Dict, List, Optional

# =============================================================================
# PARAMETERS
# =============================================================================

# T4/T5 weights (T5 dominant for dark preference - Klapoetke et al.)
T4_WEIGHT = 0.2          # Bright/ON pathway (~45%)
T5_WEIGHT = 1.0          # Dark/OFF pathway (~55%, normalized to 1.0)

# Spatial parameters
DEG_TO_PX = 4.0
RF_RADIUS_DEG = 30.0     # 60° diameter RF
CENTER_SIGMA_DEG = 6.0   # Reduced from 10° to 6° - tighter center weighting
                         # 10° bar edges at ~5° → weight ≈ 66%
                         # 60° bar edges at ~30° → weight ≈ 0%

# Multiplicative model parameters
OPPONENT_THRESHOLD = 0.0  # Threshold before rectification
MULTIPLICATION_POWER = 0.75  # Increased from 0.5 to 0.75
                             # This STRONGLY penalizes weak quadrants
                             # Looming (all 4 strong) >> bars (2 strong, 2 weak)

# Partial-match pathway - DISABLED due to grating leakage
# The preprocessing creates bilateral signals even for gratings
ADDITIVE_WEIGHT = 0.0  # Pure multiplicative only

# Narrow-bar detector - DISABLED for debugging
# Once preprocessing is fixed, this can be re-enabled
NARROW_BAR_WEIGHT = 0.0  # Disabled to isolate the problem

# Center weighting parameter (not used currently)
ADDITIVE_CENTER_SIGMA_DEG = 8.0

# Global gain (amplifies all responses)
GLOBAL_GAIN = 300.0  # Adjusted to give looming ~70-90%, bars ~20-40%

# Version marker for debugging
MODEL_VERSION = "v6.9_power_0.75"

# Output dynamics (GCaMP6f-like)
TAU_RISE = 0.050    # 50ms
TAU_DECAY = 0.300   # 300ms
SIGMOID_HALF = 100.0  # Increased from 50 - more gradual saturation curve
SIGMOID_MAX = 100.0

# Constants
FPS = 60
DT = 1.0 / FPS

# Directories
PREPROCESSED_DIR = './lplc2_preprocessed'
OUTPUT_DIR = './lplc2_output'

# Figure 2 stimuli
FIG2D_STIMULI = [
    'fig2d_dark_rv10', 'fig2d_dark_rv20', 'fig2d_dark_rv40', 'fig2d_dark_rv80',
    'fig2d_bright_rv10', 'fig2d_bright_rv20', 'fig2d_bright_rv40', 'fig2d_bright_rv80',
]

FIG2F_STIMULI = [
    'fig2f_a_looming',
    'fig2f_b_bar_rightward',
    'fig2f_c_edge_rightward',
    'fig2f_d_bar_downward',
    'fig2f_e_fullfield_downward',
    'fig2f_f_grating_rightward',
    'fig2f_g_grating_downward',
]

# Figure 2C stimuli - Expanding bars
# Tests directional selectivity with bar stimuli
FIG2C_STIMULI = [
    'fig2c_a_bar_diagonal_NE_SW',      # Single bar, NE-SW orientation, expanding
    'fig2c_b_bar_diagonal_NW_SE',      # Single bar, NW-SE orientation, expanding
    'fig2c_c_bar_horizontal',           # Single horizontal bar, expanding vertically
    'fig2c_d_cross_both_outward',       # X pattern, both bars expanding outward
    'fig2c_e_cross_opposing',           # X pattern, one outward + one inward
]

FIG2C_LABELS = [
    'Bar ↗↙ (expand)',
    'Bar ↖↘ (expand)',
    'Bar ― (expand)',
    'Cross ✕ (both out)',
    'Cross ✕ (opposing)',
]

# =============================================================================
# SPATIAL WEIGHTS
# =============================================================================

def create_quadrant_weights(height: int, width: int) -> Dict[str, np.ndarray]:
    """
    Create spatial weight maps for the four quadrants of the receptive field.

    Each quadrant has a Gaussian-weighted region on its side of the visual field.
    This matches LPLC2's cross-shaped dendritic structure.
    """
    cy, cx = height // 2, width // 2
    y_grid, x_grid = np.ogrid[:height, :width]

    # Distance from center in degrees
    dx_deg = (x_grid - cx) / DEG_TO_PX
    dy_deg = (y_grid - cy) / DEG_TO_PX
    dist_deg = np.sqrt(dx_deg**2 + dy_deg**2)

    # Base Gaussian weight (RF envelope)
    rf_weight = np.exp(-dist_deg**2 / (2 * CENTER_SIGMA_DEG**2))
    rf_weight[dist_deg > RF_RADIUS_DEG] = 0

    # Quadrant masks (smooth transitions using sigmoid-like weighting)
    # Right quadrant: x > 0
    right_mask = 1 / (1 + np.exp(-dx_deg / 3))  # Smooth transition
    # Left quadrant: x < 0
    left_mask = 1 / (1 + np.exp(dx_deg / 3))
    # Top quadrant: y < 0 (image coordinates, so negative y is up)
    top_mask = 1 / (1 + np.exp(dy_deg / 3))
    # Bottom quadrant: y > 0
    bottom_mask = 1 / (1 + np.exp(-dy_deg / 3))

    # Combine with RF envelope
    weights = {
        'right': (rf_weight * right_mask).astype(np.float32),
        'left': (rf_weight * left_mask).astype(np.float32),
        'top': (rf_weight * top_mask).astype(np.float32),
        'bottom': (rf_weight * bottom_mask).astype(np.float32),
        'full': rf_weight.astype(np.float32)
    }

    # Normalize each quadrant
    for key in weights:
        if weights[key].sum() > 0:
            weights[key] = weights[key] / weights[key].sum()

    # === CENTER WEIGHT for additive pathway (favors LOCAL stimuli) ===
    # Sharper Gaussian that weights center more heavily
    center_weight = np.exp(-dist_deg**2 / (2 * ADDITIVE_CENTER_SIGMA_DEG**2))
    center_weight[dist_deg > RF_RADIUS_DEG] = 0
    if center_weight.sum() > 0:
        center_weight = center_weight / center_weight.sum()
    weights['center'] = center_weight.astype(np.float32)

    return weights

# =============================================================================
# MODEL
# =============================================================================

def compute_directional_motion(flow: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract motion in each cardinal direction from optical flow.

    flow: (n_frames, H, W, 2) where [...,0] is vx, [...,1] is vy
    """
    vx = flow[..., 0]  # Positive = rightward
    vy = flow[..., 1]  # Positive = downward

    return {
        'rightward': np.maximum(0, vx),
        'leftward': np.maximum(0, -vx),
        'downward': np.maximum(0, vy),
        'upward': np.maximum(0, -vy),
    }


def run_model(preprocessed: Dict) -> Dict:
    """
    Run LPLC2 model with pure multiplicative radial opponency.

    The model computes opponent signals for each quadrant:
    - Right quadrant: rightward - leftward (should be + for expansion)
    - Left quadrant: leftward - rightward (should be + for expansion)
    - Top quadrant: upward - downward (should be + for expansion)
    - Bottom quadrant: downward - upward (should be + for expansion)

    Response = GLOBAL_GAIN × (R+ × L+ × T+ × B+)^0.25

    Pure multiplicative provides STRICT selectivity for looming.
    Single-axis expanding bars will NOT activate (product = 0).
    """
    on_signal = preprocessed['on_signal']
    off_signal = preprocessed['off_signal']
    flow = preprocessed['flow']

    n_frames, height, width = on_signal.shape

    # Create quadrant weights
    weights = create_quadrant_weights(height, width)

    # T4/T5 weighted motion signal
    motion = T4_WEIGHT * on_signal + T5_WEIGHT * off_signal

    # Compute directional motion
    dir_motion = compute_directional_motion(flow)

    # Compute response for each frame
    raw_response = np.zeros(n_frames, dtype=np.float32)

    # Store intermediate values for debugging
    opponent_signals = {
        'right': np.zeros(n_frames),
        'left': np.zeros(n_frames),
        'top': np.zeros(n_frames),
        'bottom': np.zeros(n_frames),
    }

    for t in range(n_frames):
        motion_t = motion[t]

        # === Compute opponent signals for each quadrant ===
        # These should be positive for OUTWARD expansion

        # Right quadrant: rightward motion - leftward motion
        # For looming: right side has rightward motion, minimal leftward → positive
        # For rightward bar: right side has rightward motion BUT so does left side → not selective
        right_outward = np.sum(dir_motion['rightward'][t] * motion_t * weights['right'])
        right_inward = np.sum(dir_motion['leftward'][t] * motion_t * weights['right'])
        R = right_outward - right_inward

        # Left quadrant: leftward motion - rightward motion
        left_outward = np.sum(dir_motion['leftward'][t] * motion_t * weights['left'])
        left_inward = np.sum(dir_motion['rightward'][t] * motion_t * weights['left'])
        L = left_outward - left_inward

        # Top quadrant: upward motion - downward motion
        top_outward = np.sum(dir_motion['upward'][t] * motion_t * weights['top'])
        top_inward = np.sum(dir_motion['downward'][t] * motion_t * weights['top'])
        T = top_outward - top_inward

        # Bottom quadrant: downward motion - upward motion
        bottom_outward = np.sum(dir_motion['downward'][t] * motion_t * weights['bottom'])
        bottom_inward = np.sum(dir_motion['upward'][t] * motion_t * weights['bottom'])
        B = bottom_outward - bottom_inward

        # Store for debugging
        opponent_signals['right'][t] = R
        opponent_signals['left'][t] = L
        opponent_signals['top'][t] = T
        opponent_signals['bottom'][t] = B

        # === Rectify opponent signals ===
        R_rect = max(0, R - OPPONENT_THRESHOLD)
        L_rect = max(0, L - OPPONENT_THRESHOLD)
        T_rect = max(0, T - OPPONENT_THRESHOLD)
        B_rect = max(0, B - OPPONENT_THRESHOLD)

        # === Multiplicative integration (highly selective for full radial expansion) ===
        # Use geometric mean (4th root of product) to keep response in reasonable range
        # This is equivalent to: (R × L × T × B)^0.25
        product = R_rect * L_rect * T_rect * B_rect
        multiplicative = product ** MULTIPLICATION_POWER

        # === Narrow-bar pathway (responds to spatially COMPACT bilateral expansion) ===
        # Uses STRICT ratio requirement (0.5) to reject gratings/translation
        # Only contributes if NARROW_BAR_WEIGHT > 0

        narrow_bar_response = 0.0
        if NARROW_BAR_WEIGHT > 0:
            STRICT_RATIO_THRESHOLD = 0.5  # Require at least 50% balance (stricter)

            # Vertical bilateral (for horizontal bars expanding up/down)
            if T_rect > 0.01 and B_rect > 0.01:
                tb_ratio = min(T_rect, B_rect) / max(T_rect, B_rect)
                if tb_ratio > STRICT_RATIO_THRESHOLD:
                    narrow_bar_response += np.sqrt(T_rect * B_rect)

            # Horizontal bilateral (for vertical bars expanding left/right)
            if R_rect > 0.01 and L_rect > 0.01:
                rl_ratio = min(R_rect, L_rect) / max(R_rect, L_rect)
                if rl_ratio > STRICT_RATIO_THRESHOLD:
                    narrow_bar_response += np.sqrt(R_rect * L_rect)

        # === Combined response ===
        raw_response[t] = GLOBAL_GAIN * (multiplicative + NARROW_BAR_WEIGHT * narrow_bar_response)

    # Sigmoid saturation
    saturated = SIGMOID_MAX * raw_response / (raw_response + SIGMOID_HALF + 1e-10)

    # Calcium dynamics
    calcium = np.zeros(n_frames, dtype=np.float32)
    for t in range(1, n_frames):
        alpha = DT / TAU_RISE if saturated[t] > calcium[t-1] else DT / TAU_DECAY
        calcium[t] = calcium[t-1] + alpha * (saturated[t] - calcium[t-1])

    time = np.arange(n_frames) * DT - 1.0

    # Debug: compute peak values for each opponent signal
    peak_R = np.max(opponent_signals['right'])
    peak_L = np.max(opponent_signals['left'])
    peak_T = np.max(opponent_signals['top'])
    peak_B = np.max(opponent_signals['bottom'])

    return {
        'raw': raw_response,
        'saturated': saturated,
        'calcium': calcium,
        'time': time,
        'peak': calcium.max(),
        'opponent_signals': opponent_signals,
        'debug': {
            'peak_R': peak_R,
            'peak_L': peak_L,
            'peak_T': peak_T,
            'peak_B': peak_B,
        }
    }

# =============================================================================
# I/O
# =============================================================================

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
        print(f"Missing required data for {name}")
        return None

    return result


def get_all_preprocessed_stimuli() -> List[str]:
    """Discover all preprocessed stimuli in the preprocessed directory."""
    if not os.path.exists(PREPROCESSED_DIR):
        return []

    stimuli = []
    for item in sorted(os.listdir(PREPROCESSED_DIR)):
        item_path = os.path.join(PREPROCESSED_DIR, item)
        if os.path.isdir(item_path):
            # Check if it has required files
            required = ['on_signal.npy', 'off_signal.npy', 'flow.npy']
            if all(os.path.exists(os.path.join(item_path, f)) for f in required):
                stimuli.append(item)

    return stimuli

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_all_stimuli_figure(results: Dict[str, Dict], output_dir: str):
    """
    Create a comprehensive figure showing ΔF/F traces for ALL preprocessed stimuli,
    arranged side by side in a grid layout.
    """
    if not results:
        print("No results to plot.")
        return

    # Sort stimuli names for consistent ordering
    stimuli_names = sorted(results.keys())
    n_stimuli = len(stimuli_names)

    if n_stimuli == 0:
        return

    # Determine grid layout (aim for roughly square, max 6 columns)
    max_cols = 6
    n_cols = min(n_stimuli, max_cols)
    n_rows = int(np.ceil(n_stimuli / n_cols))

    # Determine y-axis scale (use consistent scale across all plots)
    all_peaks = [results[s]['peak'] for s in stimuli_names]
    max_peak = max(all_peaks) if all_peaks else 100
    y_scale = max(100, max_peak * 1.1)  # At least 100, or 10% above max

    # Create figure
    fig_width = 3 * n_cols
    fig_height = 2.5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Handle single row/column cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each stimulus
    for idx, stim_name in enumerate(stimuli_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        r = results[stim_name]

        # Color coding: green for looming, blue for dark, orange for bright, gray for others
        if 'looming' in stim_name.lower():
            color = 'green'
        elif 'dark' in stim_name.lower():
            color = 'black'
        elif 'bright' in stim_name.lower():
            color = 'orange'
        else:
            color = 'steelblue'

        ax.plot(r['time'], r['calcium'], color=color, lw=1.2)
        ax.fill_between(r['time'], 0, r['calcium'], color=color, alpha=0.2)

        ax.set_xlim([r['time'][0], r['time'][-1]])
        ax.set_ylim([0, y_scale])

        # Shortened title (remove common prefixes)
        short_name = stim_name
        for prefix in ['fig2d_', 'fig2f_', 'fig2_']:
            if short_name.startswith(prefix):
                short_name = short_name[len(prefix):]
                break

        ax.set_title(f'{short_name}\n(peak: {r["peak"]:.1f}%)', fontsize=8)

        if col == 0:
            ax.set_ylabel('ΔF/F (%)', fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel('Time (s)', fontsize=8)

        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide empty subplots
    for idx in range(n_stimuli, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.suptitle(f'LPLC2 Model: ΔF/F Responses for All Stimuli (n={n_stimuli})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_stimuli_deltaF_F.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Created: all_stimuli_deltaF_F.png ({n_stimuli} stimuli, {n_rows}x{n_cols} grid)")


def create_figure_2c(results: Dict[str, Dict], output_dir: str):
    """
    Create Figure 2C: Expanding bar responses.

    Tests LPLC2 selectivity for radially expanding stimuli:
    - Single expanding bars in different orientations
    - Crossed bars (X) with both expanding outward (should respond)
    - Crossed bars with opposing motion (should suppress)
    """
    available = [(s, l) for s, l in zip(FIG2C_STIMULI, FIG2C_LABELS) if s in results]
    if not available:
        return []

    stimuli, labels = zip(*available)
    peaks = [results[s]['peak'] for s in stimuli]

    # Determine scale
    max_peak = max(peaks) if peaks else 100
    scale = max(100, max_peak * 1.1)

    n = len(available)

    # === Panel 1: Time traces side by side ===
    fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 3.5))
    if n == 1:
        axes = [axes]

    for i, (stim, label) in enumerate(available):
        r = results[stim]
        ax = axes[i]

        # Color code: green for cross-outward (expected strong), red for opposing
        if 'both_outward' in stim or 'cross_both' in stim:
            color = 'green'
        elif 'opposing' in stim:
            color = 'red'
        else:
            color = 'steelblue'

        ax.plot(r['time'], r['calcium'], color='black', lw=1.2)
        ax.fill_between(r['time'], 0, r['calcium'], color=color, alpha=0.3)
        ax.set_xlim([r['time'][0], r['time'][-1]])
        ax.set_ylim([0, scale])
        ax.set_title(f'{label}\n(peak: {r["peak"]:.1f}%)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        if i == 0:
            ax.set_ylabel('ΔF/F (%)', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Figure 2C: Expanding Bar Responses', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2c_traces.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # === Panel 2: Bar chart comparison ===
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = []
    for s in stimuli:
        if 'both_outward' in s or 'cross_both' in s:
            colors.append('green')
        elif 'opposing' in s:
            colors.append('salmon')
        else:
            colors.append('steelblue')

    bars = ax.bar(range(len(peaks)), peaks, color=colors, edgecolor='black', width=0.7)

    for bar, p in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, p + 1, f'{p:.0f}', ha='center', fontsize=10)

    ax.set_xticks(range(len(peaks)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Peak ΔF/F (%)')
    ax.set_ylim([0, max(peaks) * 1.25 if peaks else 100])

    # Check cross-bar selectivity (d vs e)
    cross_out = results.get('fig2c_d_cross_both_outward', {}).get('peak', 0)
    cross_opp = results.get('fig2c_e_cross_opposing', {}).get('peak', 1)

    if cross_out > 0 and cross_opp > 0:
        ratio = cross_out / cross_opp
        color = 'green' if ratio > 2 else 'orange' if ratio > 1.5 else 'red'
        status = '✓' if ratio > 2 else ''
        ax.set_title(f'Cross outward/opposing = {ratio:.1f}x {status}\n(Both-out should >> opposing)',
                    fontsize=11, color=color)
    else:
        ax.set_title('Figure 2C: Expanding Bar Selectivity', fontsize=11)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2c_bars.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # === Panel 3: Opponent signals for insight ===
    fig, axes = plt.subplots(2, len(available), figsize=(3 * len(available), 5))
    if len(available) == 1:
        axes = axes.reshape(-1, 1)

    for i, (stim, label) in enumerate(available):
        r = results[stim]
        opp = r['opponent_signals']
        time = r['time']

        # Top: raw opponent signals
        ax = axes[0, i]
        ax.plot(time, opp['right'], 'r-', lw=1, label='R', alpha=0.8)
        ax.plot(time, opp['left'], 'b-', lw=1, label='L', alpha=0.8)
        ax.plot(time, opp['top'], 'g-', lw=1, label='T', alpha=0.8)
        ax.plot(time, opp['bottom'], 'm-', lw=1, label='B', alpha=0.8)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_title(label, fontsize=9)
        if i == 0:
            ax.set_ylabel('Opponent', fontsize=8)
            ax.legend(fontsize=6, loc='upper left', ncol=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Bottom: rectified (what contributes to product)
        ax = axes[1, i]
        R_rect = np.maximum(0, opp['right'])
        L_rect = np.maximum(0, opp['left'])
        T_rect = np.maximum(0, opp['top'])
        B_rect = np.maximum(0, opp['bottom'])

        ax.fill_between(time, 0, R_rect, alpha=0.4, color='r')
        ax.fill_between(time, 0, L_rect, alpha=0.4, color='b')
        ax.fill_between(time, 0, T_rect, alpha=0.4, color='g')
        ax.fill_between(time, 0, B_rect, alpha=0.4, color='m')
        ax.set_xlabel('Time (s)', fontsize=8)
        if i == 0:
            ax.set_ylabel('Rectified [x]+', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Figure 2C: Opponent Signals\n(Product requires all 4 positive)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2c_opponents.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Created: figure_2c_traces.png, figure_2c_bars.png, figure_2c_opponents.png")

    return peaks


def create_figure_2d(results: Dict[str, Dict], output_dir: str):
    """Create Figure 2D: Dark vs Bright looming."""
    rv_values = [10, 20, 40, 80]

    dark_peaks = [results[f'fig2d_dark_rv{rv}']['peak']
                  for rv in rv_values if f'fig2d_dark_rv{rv}' in results]
    bright_peaks = [results[f'fig2d_bright_rv{rv}']['peak']
                    for rv in rv_values if f'fig2d_bright_rv{rv}' in results]

    all_peaks = dark_peaks + bright_peaks
    scale = 100 if not all_peaks or max(all_peaks) > 50 else 50

    # Side-by-side traces
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for i, rv in enumerate(rv_values):
        # Dark (top row)
        name = f'fig2d_dark_rv{rv}'
        if name in results:
            r = results[name]
            ax = axes[0, i]
            ax.plot(r['time'], r['calcium'], 'k-', lw=1)
            ax.fill_between(r['time'], 0, r['calcium'], color='gray', alpha=0.3)
            ax.set_xlim([r['time'][0], r['time'][-1]])
            ax.set_ylim([0, scale])
            ax.set_title(f'Dark r/v={rv}ms\n({r["peak"]:.0f}%)', fontsize=9)
            if i == 0:
                ax.set_ylabel('ΔF/F (%)', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Bright (bottom row)
        name = f'fig2d_bright_rv{rv}'
        if name in results:
            r = results[name]
            ax = axes[1, i]
            ax.plot(r['time'], r['calcium'], 'k-', lw=1)
            ax.fill_between(r['time'], 0, r['calcium'], color='lightgray', alpha=0.3)
            ax.set_xlim([r['time'][0], r['time'][-1]])
            ax.set_ylim([0, scale])
            ax.set_title(f'Bright r/v={rv}ms\n({r["peak"]:.0f}%)', fontsize=9)
            ax.set_xlabel('Time (s)', fontsize=8)
            if i == 0:
                ax.set_ylabel('ΔF/F (%)', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.suptitle('Figure 2D: Dark vs Bright Looming', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2d_traces.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    if dark_peaks:
        ax.plot(rv_values[:len(dark_peaks)], dark_peaks, 'ko-', lw=2, ms=8, label='Dark')
    if bright_peaks:
        ax.plot(rv_values[:len(bright_peaks)], bright_peaks, 's--', color='gray', lw=2, ms=8, label='Bright')
    ax.set_xscale('log')
    ax.set_xlabel('r/v (ms)')
    ax.set_ylabel('Peak ΔF/F (%)')
    ax.set_title('Peak vs r/v')
    ax.set_xlim([8, 100])
    ax.set_ylim([0, scale])
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    if dark_peaks and bright_peaks:
        dark_mean = np.mean(dark_peaks)
        bright_mean = np.mean(bright_peaks)
        ratio = bright_mean / dark_mean * 100 if dark_mean > 0 else 0

        bars = ax.bar(['Dark', 'Bright'], [dark_mean, bright_mean],
                     color=['black', 'lightgray'], edgecolor='black', width=0.5)
        for bar, val in zip(bars, [dark_mean, bright_mean]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%', ha='center')

        ax.set_ylabel('Mean Peak ΔF/F (%)')
        ax.set_ylim([0, max(dark_mean, bright_mean) * 1.3])

        color = 'green' if 20 <= ratio <= 30 else 'orange' if 15 <= ratio <= 40 else 'red'
        status = '✓' if 20 <= ratio <= 30 else ''
        ax.set_title(f'Bright/Dark = {ratio:.1f}% {status}\n(Target: 20-30%)', color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2d_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()

    return dark_peaks, bright_peaks


def create_figure_2f(results: Dict[str, Dict], output_dir: str):
    """Create Figure 2F: Selectivity test with opponent signal visualization."""
    labels = ['Looming', 'Bar →', 'Edge →', 'Bar ↓', 'Full-field ↓', 'Grating →', 'Grating ↓']

    available = [(s, l) for s, l in zip(FIG2F_STIMULI, labels) if s in results]
    if not available:
        return []

    stimuli, lbls = zip(*available)
    peaks = [results[s]['peak'] for s in stimuli]

    scale = 100 if max(peaks) > 50 else 50

    # Side-by-side traces
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3))
    if n == 1:
        axes = [axes]

    for i, (stim, label) in enumerate(available):
        r = results[stim]
        ax = axes[i]

        color = 'green' if 'looming' in stim else 'gray'
        ax.plot(r['time'], r['calcium'], 'k-', lw=1)
        ax.fill_between(r['time'], 0, r['calcium'], color=color, alpha=0.3)
        ax.set_xlim([r['time'][0], r['time'][-1]])
        ax.set_ylim([0, scale])
        ax.set_title(f'{label}\n({r["peak"]:.0f}%)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        if i == 0:
            ax.set_ylabel('ΔF/F (%)', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Figure 2F: Stimulus Selectivity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2f_traces.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['green' if 'looming' in s else 'gray' for s in stimuli]
    bars = ax.bar(range(len(peaks)), peaks, color=colors, edgecolor='black', width=0.7)

    for bar, p in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width()/2, p + 1, f'{p:.0f}', ha='center', fontsize=10)

    ax.set_xticks(range(len(peaks)))
    ax.set_xticklabels(lbls, rotation=30, ha='right')
    ax.set_ylabel('Peak ΔF/F (%)')
    ax.set_ylim([0, max(peaks) * 1.25 if peaks else 100])

    # Reference line
    if 'fig2f_a_looming' in results:
        loom_peak = results['fig2f_a_looming']['peak']
        ax.axhline(loom_peak * 0.3, color='red', ls='--', alpha=0.7, label='30% of looming')
        ax.legend(loc='upper right')

    # Selectivity
    loom = results.get('fig2f_a_looming', {}).get('peak', 0)
    others = [results[s]['peak'] for s in stimuli if 'looming' not in s]
    max_other = max(others) if others else 1
    selectivity = loom / max_other if max_other > 0 else 0

    color = 'green' if selectivity > 3 else 'orange' if selectivity > 2 else 'red'
    status = '✓' if selectivity > 3 else ''
    ax.set_title(f'Selectivity = {selectivity:.1f}x {status} (Target: >3x)', fontsize=12, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2f_bars.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # === NEW: Opponent signal visualization ===
    fig, axes = plt.subplots(2, len(available), figsize=(3 * len(available), 6))

    for i, (stim, label) in enumerate(available):
        r = results[stim]
        opp = r['opponent_signals']
        time = r['time']

        # Top row: individual opponent signals
        ax = axes[0, i]
        ax.plot(time, opp['right'], 'r-', lw=1, label='Right', alpha=0.7)
        ax.plot(time, opp['left'], 'b-', lw=1, label='Left', alpha=0.7)
        ax.plot(time, opp['top'], 'g-', lw=1, label='Top', alpha=0.7)
        ax.plot(time, opp['bottom'], 'm-', lw=1, label='Bottom', alpha=0.7)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_title(f'{label}', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        if i == 0:
            ax.set_ylabel('Opponent Signal', fontsize=8)
            ax.legend(fontsize=6, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Bottom row: product components
        ax = axes[1, i]
        R_rect = np.maximum(0, opp['right'])
        L_rect = np.maximum(0, opp['left'])
        T_rect = np.maximum(0, opp['top'])
        B_rect = np.maximum(0, opp['bottom'])

        ax.fill_between(time, 0, R_rect, alpha=0.3, label='R+', color='r')
        ax.fill_between(time, 0, L_rect, alpha=0.3, label='L+', color='b')
        ax.fill_between(time, 0, T_rect, alpha=0.3, label='T+', color='g')
        ax.fill_between(time, 0, B_rect, alpha=0.3, label='B+', color='m')
        ax.set_xlabel('Time (s)', fontsize=8)
        if i == 0:
            ax.set_ylabel('Rectified', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Opponent Signals by Quadrant\n(All 4 must be positive for response)',
                fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2f_opponents.png'), dpi=200, bbox_inches='tight')
    plt.close()

    return peaks


def create_summary(results: Dict, dark_peaks: List, bright_peaks: List, output_dir: str):
    """Create parameter summary."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')

    ratio = np.mean(bright_peaks) / np.mean(dark_peaks) * 100 if dark_peaks and bright_peaks and np.mean(dark_peaks) > 0 else 0

    loom = results.get('fig2f_a_looming', {}).get('peak', 0)
    others = [results[s]['peak'] for s in FIG2F_STIMULI[1:] if s in results]
    max_other = max(others) if others else 1
    selectivity = loom / max_other if max_other > 0 else 0

    # Figure 2C metrics
    cross_out = results.get('fig2c_d_cross_both_outward', {}).get('peak', 0)
    cross_opp = results.get('fig2c_e_cross_opposing', {}).get('peak', 1)
    cross_ratio = cross_out / cross_opp if cross_opp > 0 else 0

    text = f"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║       LPLC2 MODEL - PURE MULTIPLICATIVE RADIAL OPPONENCY               ║
    ║       Based on Klapoetke et al. 2017 & Zhao et al. 2023                ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║  MECHANISM                                                             ║
    ║  ─────────                                                             ║
    ║    Response = GAIN × (R+ × L+ × T+ × B+)^{MULTIPLICATION_POWER}                          ║
    ║    Where:                                                              ║
    ║      R+ = max(0, rightward_right - leftward_right)                     ║
    ║      L+ = max(0, leftward_left - rightward_left)                       ║
    ║      T+ = max(0, upward_top - downward_top)                            ║
    ║      B+ = max(0, downward_bottom - upward_bottom)                      ║
    ║                                                                        ║
    ║    Higher power ({MULTIPLICATION_POWER}) penalizes weak quadrants more.                  ║
    ║    Looming (all 4 strong) > Bars (2 strong, 2 weak)                    ║
    ║                                                                        ║
    ║  PARAMETERS                                                            ║
    ║  ──────────                                                            ║
    ║    T4_WEIGHT (bright):     {T4_WEIGHT:<8}                                   ║
    ║    T5_WEIGHT (dark):       {T5_WEIGHT:<8}                                   ║
    ║    RF_RADIUS:              {RF_RADIUS_DEG}°                                  ║
    ║    CENTER_SIGMA:           {CENTER_SIGMA_DEG}°                                  ║
    ║    MULTIPLICATION_POWER:   {MULTIPLICATION_POWER:<8}                                   ║
    ║    GLOBAL_GAIN:            {GLOBAL_GAIN:<8}                                   ║
    ║                                                                        ║
    ║  FIGURE 2D: DARK VS BRIGHT                                             ║
    ║  ─────────────────────────                                             ║
    ║    Dark mean peak:         {np.mean(dark_peaks) if dark_peaks else 0:.1f}%                              ║
    ║    Bright mean peak:       {np.mean(bright_peaks) if bright_peaks else 0:.1f}%                              ║
    ║    Bright/Dark ratio:      {ratio:.1f}%                               ║
    ║    Target:                 20-30%                                      ║
    ║    Status:                 {'✓ PASS' if 20 <= ratio <= 30 else '~ CLOSE' if 15 <= ratio <= 40 else '✗ FAIL'}                               ║
    ║                                                                        ║
    ║  FIGURE 2F: SELECTIVITY                                                ║
    ║  ──────────────────────                                                ║
    ║    Looming peak:           {loom:.1f}%                              ║
    ║    Max other stimulus:     {max_other:.1f}%                              ║
    ║    Selectivity ratio:      {selectivity:.1f}x                               ║
    ║    Target:                 >3x                                         ║
    ║    Status:                 {'✓ PASS' if selectivity > 3 else '~ CLOSE' if selectivity > 2 else '✗ FAIL'}                               ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """

    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontfamily='monospace',
           fontsize=9, verticalalignment='center', horizontalalignment='center')

    plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=200, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def run_figure2_analysis(verbose=True):
    """Run multiplicative opponency model on Figure 2 stimuli."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if verbose:
        print(f"LPLC2 Model - {MODEL_VERSION}")
        print("=" * 55)
        print("Based on Klapoetke et al. 2017 & Zhao et al. 2023")
        print("-" * 55)
        print(f"Mechanism: GAIN × (R+ × L+ × T+ × B+)^{MULTIPLICATION_POWER}")
        print("  Higher power penalizes weak quadrants (bars < looming)")
        print(f"  CENTER_SIGMA: {CENTER_SIGMA_DEG}°")
        print("-" * 55)
        print(f"T4/T5 weights: {T4_WEIGHT} / {T5_WEIGHT}")
        print(f"RF radius: {RF_RADIUS_DEG}°")
        print(f"Global gain: {GLOBAL_GAIN}")
        print("-" * 55)

    results = {}

    # === NEW: Process ALL stimuli in preprocessed folder ===
    all_stimuli = get_all_preprocessed_stimuli()
    if verbose:
        print(f"\nFound {len(all_stimuli)} preprocessed stimuli in {PREPROCESSED_DIR}")

    # Process all stimuli
    if verbose:
        print("\nProcessing all stimuli:")
        print(f"{'Name':<35} {'Peak':>8} {'R+':>8} {'L+':>8} {'T+':>8} {'B+':>8}")
        print("-" * 75)
    for name in all_stimuli:
        preprocessed = load_preprocessed(name)
        if preprocessed is None:
            if verbose:
                print(f"  {name}: LOAD FAILED")
            continue
        result = run_model(preprocessed)
        results[name] = result
        if verbose:
            d = result.get('debug', {})
            print(f"  {name:<33} {result['peak']:>7.1f}% {d.get('peak_R',0):>7.1f} {d.get('peak_L',0):>7.1f} {d.get('peak_T',0):>7.1f} {d.get('peak_B',0):>7.1f}")

    if not results:
        print("\nNo stimuli found! Run preprocessing first.")
        return {}

    # Create figures
    if verbose:
        print("\n" + "-" * 55)
        print("Creating figures...")

    # === NEW: Create all-stimuli ΔF/F figure ===
    create_all_stimuli_figure(results, OUTPUT_DIR)

    # Create standard Figure 2D and 2F if those stimuli exist
    dark_peaks, bright_peaks = [], []
    if any(s in results for s in FIG2D_STIMULI):
        dark_peaks, bright_peaks = create_figure_2d(results, OUTPUT_DIR)

    # Create Figure 2C (expanding bars) if those stimuli exist
    if any(s in results for s in FIG2C_STIMULI):
        create_figure_2c(results, OUTPUT_DIR)

    if any(s in results for s in FIG2F_STIMULI):
        create_figure_2f(results, OUTPUT_DIR)

    create_summary(results, dark_peaks, bright_peaks, OUTPUT_DIR)

    # Summary
    if verbose:
        print("\n" + "=" * 55)
        print("RESULTS")
        print("=" * 55)

        if dark_peaks and bright_peaks:
            ratio = np.mean(bright_peaks) / np.mean(dark_peaks) * 100
            print(f"\nFigure 2D - Bright/Dark: {ratio:.1f}% ", end='')
            print('✓' if 20 <= ratio <= 30 else f'(target: 20-30%)')

        # Figure 2C summary
        if 'fig2c_d_cross_both_outward' in results and 'fig2c_e_cross_opposing' in results:
            cross_out = results['fig2c_d_cross_both_outward']['peak']
            cross_opp = results['fig2c_e_cross_opposing']['peak']
            if cross_opp > 0:
                ratio = cross_out / cross_opp
                print(f"Figure 2C - Cross out/opposing: {ratio:.1f}x ", end='')
                print('✓' if ratio > 2 else '(target: >2x)')

        if 'fig2f_a_looming' in results:
            loom = results['fig2f_a_looming']['peak']
            others = [results[s]['peak'] for s in FIG2F_STIMULI[1:] if s in results]
            if others:
                max_other = max(others)
                if max_other > 0.01:
                    selectivity = loom / max_other
                    print(f"Figure 2F - Selectivity: {selectivity:.1f}x ", end='')
                    print('✓' if selectivity > 3 else '(target: >3x)')
                else:
                    print(f"Figure 2F - Selectivity: INFINITE (others ≈ 0) ✓")

        print(f"\nOutput: {OUTPUT_DIR}/")
        print("  - all_stimuli_deltaF_F.png  (NEW: all stimuli side-by-side)")
        print("  - figure_2c_traces.png      (expanding bars)")
        print("  - figure_2c_bars.png        (expanding bars summary)")
        print("  - figure_2c_opponents.png   (expanding bars opponent signals)")
        print("  - figure_2d_traces.png")
        print("  - figure_2d_summary.png")
        print("  - figure_2f_traces.png")
        print("  - figure_2f_bars.png")
        print("  - figure_2f_opponents.png  (shows why selectivity works)")
        print("  - summary.png")

    return results


if __name__ == "__main__":
    results = run_figure2_analysis()