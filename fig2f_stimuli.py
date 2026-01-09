"""
LPLC2 Figure 2F Stimulus Generation
====================================

These stimuli test LPLC2 selectivity for looming vs other motion types.
Only looming (2F-a) should produce a strong response.

Add these functions to your existing lplc2_stimulus_generator.py
"""

import numpy as np
from typing import Tuple

# These should match your existing constants
DEG_TO_PX = 4.0
FRAME_WIDTH = int(90 * DEG_TO_PX)   # 360 pixels
FRAME_HEIGHT = int(90 * DEG_TO_PX)  # 360 pixels
BACKGROUND = 128
DARK = 0
BRIGHT = 255
FPS = 60


def deg_to_px(degrees: float) -> float:
    """Convert degrees to pixels."""
    return degrees * DEG_TO_PX


def create_blank_frame(height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> np.ndarray:
    """Create a blank gray frame."""
    return np.full((height, width), BACKGROUND, dtype=np.uint8)


def create_coordinate_grids(height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> Tuple[np.ndarray, np.ndarray]:
    """Create coordinate grids centered on the frame."""
    y = np.arange(height) - height // 2
    x = np.arange(width) - width // 2
    X, Y = np.meshgrid(x, y)
    return X, Y


# =============================================================================
# FIGURE 2F STIMULI
# =============================================================================

def generate_fig2f_a_looming(
    duration_s: float = 5.0,
    fps: int = FPS,
    edge_speed_deg_s: float = 20.0,
    initial_size_deg: float = 5.0,
    max_size_deg: float = 60.0
) -> np.ndarray:
    """
    Figure 2F-a: Looming (expanding dark disk).

    This is the ONLY stimulus that should produce a strong LPLC2 response.
    Uses constant edge velocity for simplicity.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()
    R = np.sqrt(X**2 + Y**2)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Diameter grows at constant edge speed
        diameter_deg = initial_size_deg + edge_speed_deg_s * t
        diameter_deg = min(diameter_deg, max_size_deg)
        radius_px = deg_to_px(diameter_deg / 2)

        frame = create_blank_frame()
        frame[R <= radius_px] = DARK
        frames[t_idx] = frame

    return frames


def generate_fig2f_b_bar_rightward(
    duration_s: float = 5.0,
    fps: int = FPS,
    bar_width_deg: float = 10.0,
    bar_height_deg: float = 60.0,
    speed_deg_s: float = 20.0,
    start_x_deg: float = -45.0
) -> np.ndarray:
    """
    Figure 2F-b: Single vertical dark bar translating rightward.

    Wide-field translational motion - should NOT activate LPLC2.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    bar_half_width_px = deg_to_px(bar_width_deg / 2)
    bar_half_height_px = deg_to_px(bar_height_deg / 2)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Bar center position (moving rightward)
        center_x_deg = start_x_deg + speed_deg_s * t
        center_x_px = deg_to_px(center_x_deg)

        frame = create_blank_frame()

        # Vertical bar
        in_width = np.abs(X - center_x_px) <= bar_half_width_px
        in_height = np.abs(Y) <= bar_half_height_px

        frame[in_width & in_height] = DARK
        frames[t_idx] = frame

    return frames


def generate_fig2f_c_edge_rightward(
    duration_s: float = 5.0,
    fps: int = FPS,
    speed_deg_s: float = 20.0,
    start_x_deg: float = -45.0
) -> np.ndarray:
    """
    Figure 2F-c: Vertical edge (dark on left) translating rightward.

    Full-field edge motion - should NOT activate LPLC2.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Edge position (moving rightward)
        edge_x_deg = start_x_deg + speed_deg_s * t
        edge_x_px = deg_to_px(edge_x_deg)

        frame = create_blank_frame()

        # Dark region to the left of edge
        frame[X <= edge_x_px] = DARK

        frames[t_idx] = frame

    return frames


def generate_fig2f_d_bar_downward(
    duration_s: float = 5.0,
    fps: int = FPS,
    bar_width_deg: float = 60.0,
    bar_height_deg: float = 10.0,
    speed_deg_s: float = 20.0,
    start_y_deg: float = -45.0
) -> np.ndarray:
    """
    Figure 2F-d: Single horizontal dark bar translating downward.

    Wide-field translational motion - should NOT activate LPLC2.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    bar_half_width_px = deg_to_px(bar_width_deg / 2)
    bar_half_height_px = deg_to_px(bar_height_deg / 2)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Bar center position (moving downward = positive Y)
        center_y_deg = start_y_deg + speed_deg_s * t
        center_y_px = deg_to_px(center_y_deg)

        frame = create_blank_frame()

        # Horizontal bar
        in_width = np.abs(X) <= bar_half_width_px
        in_height = np.abs(Y - center_y_px) <= bar_half_height_px

        frame[in_width & in_height] = DARK
        frames[t_idx] = frame

    return frames


def generate_fig2f_e_fullfield_downward(
    duration_s: float = 5.0,
    fps: int = FPS,
    speed_deg_s: float = 20.0,
    start_y_deg: float = -45.0
) -> np.ndarray:
    """
    Figure 2F-e: Full-field downward motion.

    A horizontal edge (dark above, gray below) that sweeps downward.
    Uniform wide-field motion - should NOT activate LPLC2.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Edge position (moving downward = positive Y)
        edge_y_deg = start_y_deg + speed_deg_s * t
        edge_y_px = deg_to_px(edge_y_deg)

        frame = create_blank_frame()

        # Dark region above the edge
        frame[Y <= edge_y_px] = DARK

        frames[t_idx] = frame

    return frames


def generate_fig2f_f_grating_rightward(
    duration_s: float = 5.0,
    fps: int = FPS,
    bar_width_deg: float = 10.0,
    speed_deg_s: float = 20.0
) -> np.ndarray:
    """
    Figure 2F-f: Vertical grating (stripes) translating rightward.

    Wide-field periodic motion - should NOT activate LPLC2.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    bar_width_px = deg_to_px(bar_width_deg)
    period_px = bar_width_px * 2  # Full period = dark + gray

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Phase offset due to motion (rightward = positive X direction)
        offset_px = deg_to_px(speed_deg_s * t)

        # Create vertical grating (vertical stripes)
        phase = (X + offset_px) % period_px

        frame = create_blank_frame()
        frame[phase < bar_width_px] = DARK

        frames[t_idx] = frame

    return frames


def generate_fig2f_g_grating_downward(
    duration_s: float = 5.0,
    fps: int = FPS,
    bar_width_deg: float = 10.0,
    speed_deg_s: float = 20.0
) -> np.ndarray:
    """
    Figure 2F-g: Horizontal grating (stripes) translating downward.

    Wide-field periodic motion - should NOT activate LPLC2.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    bar_width_px = deg_to_px(bar_width_deg)
    period_px = bar_width_px * 2  # Full period = dark + gray

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Phase offset due to motion (downward = positive Y direction)
        offset_px = deg_to_px(speed_deg_s * t)

        # Create horizontal grating (horizontal stripes)
        phase = (Y + offset_px) % period_px

        frame = create_blank_frame()
        frame[phase < bar_width_px] = DARK

        frames[t_idx] = frame

    return frames


# =============================================================================
# BATCH GENERATION FOR FIGURE 2F
# =============================================================================

def generate_all_fig2f_stimuli(
    duration_s: float = 5.0,
    fps: int = FPS,
    speed_deg_s: float = 20.0
) -> dict:
    """
    Generate all Figure 2F stimuli as a dictionary.

    Returns:
        dict mapping stimulus names to frame arrays
    """
    stimuli = {}

    print("Generating Figure 2F stimuli...")

    # 2F-a: Looming (the only one that should activate LPLC2)
    print("  fig2f_a_looming")
    stimuli['fig2f_a_looming'] = generate_fig2f_a_looming(
        duration_s=duration_s, fps=fps, edge_speed_deg_s=speed_deg_s
    )

    # 2F-b: Bar translating rightward
    print("  fig2f_b_bar_rightward")
    stimuli['fig2f_b_bar_rightward'] = generate_fig2f_b_bar_rightward(
        duration_s=duration_s, fps=fps, speed_deg_s=speed_deg_s
    )

    # 2F-c: Edge translating rightward
    print("  fig2f_c_edge_rightward")
    stimuli['fig2f_c_edge_rightward'] = generate_fig2f_c_edge_rightward(
        duration_s=duration_s, fps=fps, speed_deg_s=speed_deg_s
    )

    # 2F-d: Bar translating downward
    print("  fig2f_d_bar_downward")
    stimuli['fig2f_d_bar_downward'] = generate_fig2f_d_bar_downward(
        duration_s=duration_s, fps=fps, speed_deg_s=speed_deg_s
    )

    # 2F-e: Full-field downward motion
    print("  fig2f_e_fullfield_downward")
    stimuli['fig2f_e_fullfield_downward'] = generate_fig2f_e_fullfield_downward(
        duration_s=duration_s, fps=fps, speed_deg_s=speed_deg_s
    )

    # 2F-f: Vertical grating moving rightward
    print("  fig2f_f_grating_rightward")
    stimuli['fig2f_f_grating_rightward'] = generate_fig2f_f_grating_rightward(
        duration_s=duration_s, fps=fps, speed_deg_s=speed_deg_s
    )

    # 2F-g: Horizontal grating moving downward
    print("  fig2f_g_grating_downward")
    stimuli['fig2f_g_grating_downward'] = generate_fig2f_g_grating_downward(
        duration_s=duration_s, fps=fps, speed_deg_s=speed_deg_s
    )

    print(f"Generated {len(stimuli)} stimuli")
    return stimuli


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_fig2f_stimuli(stimuli: dict, save_dir: str = './fig2f_viz') -> None:
    """Create visualization of all Figure 2F stimuli."""
    import matplotlib.pyplot as plt
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Create summary figure
    fig, axes = plt.subplots(2, 7, figsize=(21, 6))

    stim_names = [
        'fig2f_a_looming',
        'fig2f_b_bar_rightward',
        'fig2f_c_edge_rightward',
        'fig2f_d_bar_downward',
        'fig2f_e_fullfield_downward',
        'fig2f_f_grating_rightward',
        'fig2f_g_grating_downward'
    ]

    labels = ['a: Looming', 'b: Bar →', 'c: Edge →', 'd: Bar ↓',
              'e: Full-field ↓', 'f: Grating →', 'g: Grating ↓']

    for col, (name, label) in enumerate(zip(stim_names, labels)):
        if name not in stimuli:
            continue

        frames = stimuli[name]
        n_frames = len(frames)

        # Early frame
        early_idx = n_frames // 4
        axes[0, col].imshow(frames[early_idx], cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(f'{label}\nt={early_idx/FPS:.1f}s', fontsize=10)
        axes[0, col].axis('off')

        # Add RF circle
        circle = plt.Circle((FRAME_WIDTH//2, FRAME_HEIGHT//2),
                           deg_to_px(30), fill=False, color='red',
                           linestyle='--', linewidth=1)
        axes[0, col].add_patch(circle)

        # Late frame
        late_idx = 3 * n_frames // 4
        axes[1, col].imshow(frames[late_idx], cmap='gray', vmin=0, vmax=255)
        axes[1, col].set_title(f't={late_idx/FPS:.1f}s', fontsize=10)
        axes[1, col].axis('off')

        circle = plt.Circle((FRAME_WIDTH//2, FRAME_HEIGHT//2),
                           deg_to_px(30), fill=False, color='red',
                           linestyle='--', linewidth=1)
        axes[1, col].add_patch(circle)

    plt.suptitle('Figure 2F: LPLC2 Selectivity Test Stimuli\n(Red circle = 60° receptive field)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'fig2f_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Figure 2F Stimulus Generator")
    print("=" * 60)

    # Output directories
    viz_dir = './fig2f_viz'
    npy_dir = './lplc2_inputs'
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    # Generate all stimuli
    stimuli = generate_all_fig2f_stimuli(duration_s=5.0)

    # Save as numpy arrays
    print("\nSaving numpy arrays...")
    for name, frames in stimuli.items():
        save_path = os.path.join(npy_dir, f'{name}.npy')
        np.save(save_path, frames)
        print(f"  {save_path}: {frames.shape}")

    # Create visualization
    print("\nCreating visualization...")
    visualize_fig2f_stimuli(stimuli, save_dir=viz_dir)

    print("\n" + "=" * 60)
    print("Figure 2F stimuli generated!")
    print(f"  NumPy arrays: {npy_dir}/")
    print(f"  Visualization: {viz_dir}/")
    print("\nExpected LPLC2 responses:")
    print("  fig2f_a_looming:              HIGH (looming = radial expansion)")
    print("  fig2f_b_bar_rightward:        LOW  (single bar translation)")
    print("  fig2f_c_edge_rightward:       LOW  (full-field edge motion)")
    print("  fig2f_d_bar_downward:         LOW  (single bar translation)")
    print("  fig2f_e_fullfield_downward:   LOW  (full-field uniform motion)")
    print("  fig2f_f_grating_rightward:    LOW  (periodic wide-field motion)")
    print("  fig2f_g_grating_downward:     LOW  (periodic wide-field motion)")
    print("=" * 60)