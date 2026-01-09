"""
LPLC2 Model Preprocessing Pipeline (Memory-Efficient Version)
==============================================================

Changes from original:
1. Uses float32 instead of float64 (half the memory)
2. Skips T4/T5 directional channels by default (not needed for LPLC2 model)
3. Clears memory between stimuli with garbage collection
4. Processes one stimulus at a time without keeping all in memory
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import gc
from glob import glob
from typing import Dict, Tuple, Optional, List

# =============================================================================
# CONSTANTS
# =============================================================================

DEG_TO_PX = 4.0
FPS = 60
FRAME_HEIGHT = FRAME_WIDTH = int(90 * DEG_TO_PX)  # 360 pixels

# T4/T5 pathway weights (from Klapoetke et al.)
T4_WEIGHT = 1.0   # ON pathway (bright edges)
T5_WEIGHT = 2.0   # OFF pathway (dark edges) - 2x stronger

# Noise for aperture problem mitigation
NOISE_SIGMA = 3.0
NOISE_SEED = 42

# Directories
INPUT_DIR = './lplc2_inputs'
OUTPUT_DIR = './lplc2_preprocessed'
VIZ_DIR = './lplc2_preprocessed_viz'

# Cardinal directions for T4/T5 channels
DIRECTIONS = {
    'rightward': 0,
    'leftward': 180,
    'downward': 90,
    'upward': 270
}


# =============================================================================
# CORE PREPROCESSING FUNCTIONS
# =============================================================================

def add_noise(frames: np.ndarray, sigma: float = NOISE_SIGMA, seed: Optional[int] = NOISE_SEED) -> np.ndarray:
    """Add Gaussian noise to frames to mitigate aperture problem."""
    if sigma <= 0:
        return frames.copy()

    if seed is not None:
        np.random.seed(seed)

    # Process frame by frame to save memory
    noisy = np.zeros_like(frames)
    for i in range(len(frames)):
        frame = frames[i].astype(np.float32)
        noise = np.random.randn(*frame.shape).astype(np.float32) * sigma
        noisy[i] = np.clip(frame + noise, 0, 255).astype(np.uint8)

    return noisy


def compute_optical_flow(frames: np.ndarray, winsize: int = 15) -> np.ndarray:
    """Compute dense optical flow using Farneback algorithm."""
    n_frames, height, width = frames.shape
    # Use float32 instead of default float64
    flow = np.zeros((n_frames - 1, height, width, 2), dtype=np.float32)

    for i in range(n_frames - 1):
        flow[i] = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None,
            pyr_scale=0.5,
            levels=3,
            winsize=winsize,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    return flow


def compute_on_off_signals(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ON and OFF signals from temporal luminance changes."""
    # Use float32 instead of float64
    dt = np.diff(frames.astype(np.float32), axis=0)

    on_signal = np.maximum(0, dt)
    off_signal = np.maximum(0, -dt)

    return on_signal, off_signal


def compute_t4t5_channels(
    flow: np.ndarray,
    on_signal: np.ndarray,
    off_signal: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute 8-channel T4/T5 representation.

    NOTE: This is memory-intensive. Only use if you need directional channels.
    The LPLC2 model only needs motion and radial components.
    """
    channels = {}

    for direction, angle_deg in DIRECTIONS.items():
        theta = np.radians(angle_deg)

        # Project flow onto this direction
        flow_projection = (flow[..., 0] * np.cos(theta) + flow[..., 1] * np.sin(theta)).astype(np.float32)

        # Rectify to get motion in this direction only
        directional_motion = np.maximum(0, flow_projection)

        # T4: ON pathway - use float32
        channels[f't4_{direction}'] = (on_signal * directional_motion * T4_WEIGHT).astype(np.float32)

        # T5: OFF pathway - use float32
        channels[f't5_{direction}'] = (off_signal * directional_motion * T5_WEIGHT).astype(np.float32)

        # Clear intermediate
        del flow_projection, directional_motion
        gc.collect()

    return channels


def compute_motion_energy(
    flow: np.ndarray,
    on_signal: np.ndarray,
    off_signal: np.ndarray
) -> np.ndarray:
    """Compute combined T4/T5 motion energy signal."""
    # Motion magnitude - keep as float32
    motion_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # Weight by ON/OFF signals and T4/T5 weights
    motion = (on_signal * motion_mag * T4_WEIGHT + off_signal * motion_mag * T5_WEIGHT).astype(np.float32)

    return motion


def compute_radial_motion(flow: np.ndarray) -> Dict[str, np.ndarray]:
    """Project optical flow onto radial direction from center."""
    h, w = flow.shape[1:3]
    cy, cx = h // 2, w // 2

    # Create coordinate grids - use float32
    y, x = np.ogrid[:h, :w]
    dx = (x - cx).astype(np.float32)
    dy = (y - cy).astype(np.float32)

    # Distance from center
    r = np.maximum(np.sqrt(dx**2 + dy**2), 1e-6)

    # Unit radial vectors
    rx = dx / r
    ry = dy / r

    # Project flow onto radial direction
    radial_component = (flow[..., 0] * rx + flow[..., 1] * ry).astype(np.float32)

    return {
        'outward': np.maximum(0, radial_component).astype(np.float32),
        'inward': np.maximum(0, -radial_component).astype(np.float32),
        'net': radial_component
    }


def preprocess_stimulus(
    frames: np.ndarray,
    noise_sigma: float = NOISE_SIGMA,
    noise_seed: Optional[int] = NOISE_SEED,
    compute_t4t5: bool = False,  # OFF by default to save memory
    verbose: bool = True
) -> Dict:
    """
    Run full preprocessing pipeline on a stimulus.

    Args:
        frames: Input frames
        noise_sigma: Noise level
        noise_seed: Random seed
        compute_t4t5: Whether to compute T4/T5 directional channels (memory-intensive)
        verbose: Print progress
    """
    if verbose:
        print(f"    Input shape: {frames.shape}")

    # Step 1: Add noise
    if verbose and noise_sigma > 0:
        print(f"    Adding noise (σ={noise_sigma})...")
    frames_noisy = add_noise(frames, sigma=noise_sigma, seed=noise_seed)

    # Step 2: Compute optical flow
    if verbose:
        print("    Computing optical flow...")
    flow = compute_optical_flow(frames_noisy)

    # Free noisy frames
    del frames_noisy
    gc.collect()

    # Step 3: Compute ON/OFF signals from ORIGINAL frames
    if verbose:
        print("    Computing ON/OFF signals...")
    on_signal, off_signal = compute_on_off_signals(frames)

    # Step 4: Compute combined motion energy
    if verbose:
        print("    Computing motion energy...")
    motion = compute_motion_energy(flow, on_signal, off_signal)

    # Step 5: Compute radial motion components
    if verbose:
        print("    Computing radial motion...")
    radial = compute_radial_motion(flow)

    # Step 6: Optionally compute T4/T5 channels (memory-intensive)
    t4t5 = None
    if compute_t4t5:
        if verbose:
            print("    Computing T4/T5 directional channels...")
        t4t5 = compute_t4t5_channels(flow, on_signal, off_signal)

    if verbose:
        print(f"    Output: {flow.shape[0]} frames of preprocessed data")

    result = {
        'flow': flow,
        'on_signal': on_signal,
        'off_signal': off_signal,
        'motion': motion,
        'radial': radial
    }

    if t4t5 is not None:
        result['t4t5'] = t4t5

    return result


# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def list_available_stimuli(input_dir: str = INPUT_DIR) -> List[str]:
    """List all available stimulus files in the input directory."""
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return []

    files = glob(os.path.join(input_dir, '*.npy'))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return sorted(names)


def load_stimulus(name: str, input_dir: str = INPUT_DIR) -> Optional[np.ndarray]:
    """Load a stimulus by name from the input directory."""
    path = os.path.join(input_dir, f'{name}.npy')

    if not os.path.exists(path):
        print(f"Stimulus not found: {path}")
        return None

    frames = np.load(path)
    return frames


def save_preprocessed(
    name: str,
    preprocessed: Dict,
    output_dir: str = OUTPUT_DIR
) -> None:
    """Save preprocessed data to disk."""
    stim_dir = os.path.join(output_dir, name)
    os.makedirs(stim_dir, exist_ok=True)

    # Save main arrays
    np.save(os.path.join(stim_dir, 'flow.npy'), preprocessed['flow'])
    np.save(os.path.join(stim_dir, 'on_signal.npy'), preprocessed['on_signal'])
    np.save(os.path.join(stim_dir, 'off_signal.npy'), preprocessed['off_signal'])
    np.save(os.path.join(stim_dir, 'motion.npy'), preprocessed['motion'])

    # Save T4/T5 channels if present
    if 't4t5' in preprocessed and preprocessed['t4t5'] is not None:
        t4t5_dir = os.path.join(stim_dir, 't4t5')
        os.makedirs(t4t5_dir, exist_ok=True)
        for channel_name, channel_data in preprocessed['t4t5'].items():
            np.save(os.path.join(t4t5_dir, f'{channel_name}.npy'), channel_data)

    # Save radial components
    radial_dir = os.path.join(stim_dir, 'radial')
    os.makedirs(radial_dir, exist_ok=True)
    for component_name, component_data in preprocessed['radial'].items():
        np.save(os.path.join(radial_dir, f'{component_name}.npy'), component_data)

    print(f"    Saved to: {stim_dir}/")


def load_preprocessed(name: str, output_dir: str = OUTPUT_DIR) -> Optional[Dict]:
    """Load preprocessed data from disk."""
    stim_dir = os.path.join(output_dir, name)

    if not os.path.exists(stim_dir):
        return None

    result = {}

    # Load main arrays
    for key in ['flow', 'on_signal', 'off_signal', 'motion']:
        path = os.path.join(stim_dir, f'{key}.npy')
        if os.path.exists(path):
            result[key] = np.load(path)

    # Load T4/T5 channels
    t4t5_dir = os.path.join(stim_dir, 't4t5')
    if os.path.exists(t4t5_dir):
        result['t4t5'] = {}
        for f in glob(os.path.join(t4t5_dir, '*.npy')):
            channel_name = os.path.splitext(os.path.basename(f))[0]
            result['t4t5'][channel_name] = np.load(f)

    # Load radial components
    radial_dir = os.path.join(stim_dir, 'radial')
    if os.path.exists(radial_dir):
        result['radial'] = {}
        for f in glob(os.path.join(radial_dir, '*.npy')):
            component_name = os.path.splitext(os.path.basename(f))[0]
            result['radial'][component_name] = np.load(f)

    return result


# =============================================================================
# VISUALIZATION (simplified to save memory)
# =============================================================================

def visualize_preprocessed(
    name: str,
    preprocessed: Dict,
    frames: np.ndarray,
    viz_dir: str = VIZ_DIR
) -> None:
    """Create simple visualization of preprocessed data."""
    os.makedirs(viz_dir, exist_ok=True)

    n_flow = preprocessed['flow'].shape[0]
    viz_frames = [n_flow // 6, n_flow // 2, 5 * n_flow // 6]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for col, fi in enumerate(viz_frames):
        fi = min(fi, n_flow - 1)

        # Row 0: Original frame
        axes[0, col].imshow(frames[fi], cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(f'Frame {fi}')
        axes[0, col].axis('off')

        # Row 1: Motion energy
        axes[1, col].imshow(preprocessed['motion'][fi], cmap='hot')
        axes[1, col].set_title('Motion Energy')
        axes[1, col].axis('off')

        # Row 2: Radial (outward - inward)
        radial_diff = preprocessed['radial']['outward'][fi] - preprocessed['radial']['inward'][fi]
        vmax = max(abs(radial_diff.min()), abs(radial_diff.max()), 0.1)
        axes[2, col].imshow(radial_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2, col].set_title('Radial (out-in)')
        axes[2, col].axis('off')

    plt.suptitle(f'{name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{name}_summary.png'), dpi=100, bbox_inches='tight')
    plt.close()

    # Clear figure memory
    gc.collect()


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def preprocess_all_stimuli(
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR,
    viz_dir: str = VIZ_DIR,
    save_results: bool = True,
    save_visualizations: bool = True,
    compute_t4t5: bool = False,  # OFF by default
    verbose: bool = True
) -> int:
    """
    Preprocess all stimuli in the input directory.

    Returns count of processed stimuli (does NOT keep all in memory).
    """
    stimuli = list_available_stimuli(input_dir)

    if not stimuli:
        print(f"No stimuli found in {input_dir}")
        return 0

    if verbose:
        print("=" * 70)
        print("LPLC2 Preprocessing Pipeline (Memory-Efficient)")
        print("=" * 70)
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Found {len(stimuli)} stimuli to process")
        print(f"T4 weight: {T4_WEIGHT}, T5 weight: {T5_WEIGHT}")
        print(f"Compute T4/T5 channels: {compute_t4t5}")
        print("-" * 70)

    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    if save_visualizations:
        os.makedirs(viz_dir, exist_ok=True)

    processed_count = 0

    for i, name in enumerate(stimuli):
        if verbose:
            print(f"\n[{i+1}/{len(stimuli)}] {name}")

        try:
            # Load stimulus
            frames = load_stimulus(name, input_dir)
            if frames is None:
                continue

            if verbose:
                print(f"    Loaded: {frames.shape[0]} frames")

            # Preprocess
            preprocessed = preprocess_stimulus(
                frames,
                compute_t4t5=compute_t4t5,
                verbose=verbose
            )

            # Save preprocessed data
            if save_results:
                save_preprocessed(name, preprocessed, output_dir)

            # Save visualization
            if save_visualizations:
                visualize_preprocessed(name, preprocessed, frames, viz_dir)
                if verbose:
                    print(f"    Saved visualization")

            processed_count += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        finally:
            # CRITICAL: Clear memory after each stimulus
            del frames
            if 'preprocessed' in dir():
                del preprocessed
            gc.collect()

    if verbose:
        print("\n" + "=" * 70)
        print(f"Preprocessed {processed_count}/{len(stimuli)} stimuli")
        if save_results:
            print(f"Results saved to: {os.path.abspath(output_dir)}")
        if save_visualizations:
            print(f"Visualizations saved to: {os.path.abspath(viz_dir)}")
        print("=" * 70)

    return processed_count


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Preprocess all stimuli (memory-efficient mode)
    count = preprocess_all_stimuli(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        viz_dir=VIZ_DIR,
        save_results=True,
        save_visualizations=True,
        compute_t4t5=False,  # Set to True only if you need directional channels
        verbose=True
    )

    print(f"\nDone! Processed {count} stimuli.")