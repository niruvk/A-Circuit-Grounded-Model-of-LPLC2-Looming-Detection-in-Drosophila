"""
LPLC2 Visual Stimulus Generator
Based on Klapoetke et al. 2017, Nature

This module generates visual stimuli matching those used in the paper to test LPLC2 neurons.
Each stimulus function returns numpy arrays of frames (T, H, W) with values 0-255.

Key Parameters from Paper:
- Visual field: 90° x 90° (azimuth x elevation)
- Receptive field diameter: ~60°
- Background intensity: 50% (128 gray)
- Dark stimuli: 0% intensity (0)
- Bright stimuli: 100% intensity (255)
- Frame rate: 180 Hz (paper), we use 60 Hz default

Author: Generated for LPLC2 model replication
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge, Rectangle
from typing import Tuple, List, Optional, Dict, Any
import os

# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

# Degrees to pixels conversion (adjust based on your display setup)
DEG_TO_PX = 4.0  # 4 pixels per degree

# Frame dimensions (covering 90° x 90° visual field)
FRAME_WIDTH = int(90 * DEG_TO_PX)   # 360 pixels
FRAME_HEIGHT = int(90 * DEG_TO_PX)  # 360 pixels

# Intensity values
BACKGROUND = 128    # 50% gray
DARK = 0            # Black
BRIGHT = 255        # White

# Default frame rate
FPS = 60

# Receptive field parameters
RF_DIAMETER = 60  # degrees
RF_RADIUS = RF_DIAMETER / 2  # 30 degrees


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
# FIGURE 2D: LOOMING STIMULI (r/v parameterization)
# =============================================================================

def generate_looming_rv(
    rv_ms: float,
    duration_s: float = 4.0,
    fps: int = FPS,
    final_size_deg: float = 60.0,
    initial_size_deg: float = 5.0,
    dark: bool = True,
    is_receding: bool = False,
    center_x_deg: float = 0.0,
    center_y_deg: float = 0.0
) -> np.ndarray:
    """
    Generate looming stimulus with r/v parameterization.

    The r/v parameter determines the approach speed:
    - θ(t) = 2 * arctan(r / (v * (t_collision - t)))
    - r/v = time to collision when θ = 90° (half-angle = 45°)

    Parameters:
    -----------
    rv_ms : float
        r/v ratio in milliseconds (10, 40, 80, 160, 320, 1000 ms)
    duration_s : float
        Total stimulus duration in seconds
    fps : int
        Frame rate
    final_size_deg : float
        Final disk diameter in degrees
    initial_size_deg : float
        Initial disk diameter in degrees
    dark : bool
        If True, dark disk on gray background. If False, bright disk.
    is_receding : bool
        If True, reverse the looming (contraction)
    center_x_deg, center_y_deg : float
        Center position in degrees from screen center

    Returns:
    --------
    frames : np.ndarray
        Array of shape (n_frames, height, width) with dtype uint8
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    # Convert center to pixels
    cx = deg_to_px(center_x_deg)
    cy = deg_to_px(center_y_deg)

    # Distance from center
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Convert r/v to seconds
    rv_s = rv_ms / 1000.0

    # Compute disk size at each frame
    # θ(t) = 2 * arctan(l / (v * (t_collision - t)))
    # At t=0, we want initial_size_deg
    # At t=duration_s, we want final_size_deg

    # We'll use a simplified approach: map time to diameter
    initial_radius_px = deg_to_px(initial_size_deg / 2)
    final_radius_px = deg_to_px(final_size_deg / 2)

    # For r/v looming, the expansion accelerates toward collision
    # We use the inverse tangent relationship
    for t_idx in range(n_frames):
        t = t_idx / fps

        # Normalized time (0 to 1)
        t_norm = t / duration_s

        # r/v parameterized size (faster approach = faster expansion)
        # θ = 2 * arctan(1 / ((1 - t_norm) * rv_factor))
        # Simplified: use exponential-like growth
        rv_factor = 1000 / rv_ms  # Smaller r/v = faster

        # Size grows faster as we approach "collision"
        growth = 1 - np.exp(-t_norm * rv_factor * 2)

        radius_px = initial_radius_px + (final_radius_px - initial_radius_px) * growth

        if is_receding:
            radius_px = final_radius_px + initial_radius_px - radius_px

        # Create frame
        frame = create_blank_frame()

        # Draw disk
        mask = R <= radius_px
        frame[mask] = DARK if dark else BRIGHT

        frames[t_idx] = frame

    return frames


def generate_luminance_control(
    duration_s: float = 4.0,
    fps: int = FPS,
    final_size_deg: float = 60.0,
    initial_size_deg: float = 5.0,
    rf_diameter_deg: float = 60.0
) -> np.ndarray:
    """
    Generate luminance-matched control stimulus (no motion).

    This creates a uniform darkening that matches the average luminance
    change of the looming stimulus, but without any edges or motion.

    From paper: I_disk = (D² - d²) * I_bg + d² * I_fg) / D²
    where D is RF diameter, d is disk diameter
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    D = rf_diameter_deg  # RF diameter in degrees

    for t_idx in range(n_frames):
        t_norm = t_idx / n_frames

        # Current disk diameter (if it were looming)
        d = initial_size_deg + (final_size_deg - initial_size_deg) * t_norm

        # Compute luminance-matched intensity
        # I = (D² - d²) * I_bg + d² * I_fg) / D²
        I_bg = BACKGROUND
        I_fg = DARK

        intensity = ((D**2 - d**2) * I_bg + d**2 * I_fg) / D**2

        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), int(intensity), dtype=np.uint8)
        frames[t_idx] = frame

    return frames


# =============================================================================
# FIGURE 2F: WIDE-FIELD MOTION STIMULI
# =============================================================================

def generate_constant_velocity_looming(
    edge_speed_deg_s: float = 10.0,
    duration_s: float = 5.0,
    fps: int = FPS,
    initial_size_deg: float = 5.0,
    max_size_deg: float = 60.0,
    dark: bool = True
) -> np.ndarray:
    """
    Generate constant edge velocity looming.

    Unlike r/v looming, the edge expands at constant speed.
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
        mask = R <= radius_px
        frame[mask] = DARK if dark else BRIGHT

        frames[t_idx] = frame

    return frames


def generate_widefield_translation(
    direction_deg: float,
    edge_speed_deg_s: float = 20.0,
    bar_width_deg: float = 10.0,
    duration_s: float = 5.0,
    fps: int = FPS
) -> np.ndarray:
    """
    Generate wide-field translational motion (grating).

    Parameters:
    -----------
    direction_deg : float
        Direction of motion in degrees (0 = rightward, 90 = downward)
    edge_speed_deg_s : float
        Speed of edge motion in degrees/second
    bar_width_deg : float
        Width of each bar in degrees
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    # Convert direction to radians
    theta = np.radians(direction_deg)

    # Project coordinates onto motion direction
    # This gives position along the direction of motion
    pos = X * np.cos(theta) + Y * np.sin(theta)

    bar_width_px = deg_to_px(bar_width_deg)
    period_px = bar_width_px * 2  # Full period = dark + bright

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Phase offset due to motion
        offset_px = deg_to_px(edge_speed_deg_s * t)

        # Create grating
        phase = (pos + offset_px) % period_px

        frame = create_blank_frame()
        frame[phase < bar_width_px] = DARK

        frames[t_idx] = frame

    return frames


def generate_rotation(
    direction: int = 1,  # 1 for CW, -1 for CCW
    angular_speed_deg_s: float = 20.0,
    n_spokes: int = 8,
    duration_s: float = 5.0,
    fps: int = FPS
) -> np.ndarray:
    """
    Generate rotational motion stimulus.

    Parameters:
    -----------
    direction : int
        1 for clockwise, -1 for counter-clockwise
    angular_speed_deg_s : float
        Rotational speed in degrees/second
    n_spokes : int
        Number of dark/bright pairs around the circle
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    # Compute angle for each pixel
    angles = np.arctan2(Y, X)  # -π to π

    spoke_width = np.pi / n_spokes  # Half period

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Rotation angle
        rotation = np.radians(direction * angular_speed_deg_s * t)

        # Shifted angles
        shifted_angles = (angles + rotation) % (2 * np.pi)

        # Create spoke pattern
        spoke_phase = (shifted_angles % (2 * spoke_width)) / spoke_width

        frame = create_blank_frame()
        frame[spoke_phase < 0.5] = DARK

        frames[t_idx] = frame

    return frames


# =============================================================================
# FIGURE 3E/F: BAR EXPANSION STIMULI
# =============================================================================

def generate_bar_expansion(
    orientation_deg: float,
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_separation_deg: float = 0.0,
    max_expansion_deg: float = 60.0,
    dark: bool = True
) -> np.ndarray:
    """
    Generate expanding bar stimulus.

    Two edges of a bar expand outward from the center along a specified axis.

    Parameters:
    -----------
    orientation_deg : float
        Orientation of expansion axis (0 = horizontal expansion, 90 = vertical)
    bar_width_deg : float
        Width of the bar perpendicular to expansion direction
    edge_speed_deg_s : float
        Speed of each edge in degrees/second
    duration_s : float
        Stimulus duration
    initial_separation_deg : float
        Initial separation between edges (0 = edges start at center)
    max_expansion_deg : float
        Maximum distance from center for each edge
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    # Rotate coordinate system by orientation
    theta = np.radians(orientation_deg)

    # Position along expansion axis
    pos_along = X * np.cos(theta) + Y * np.sin(theta)

    # Position perpendicular to expansion axis
    pos_perp = -X * np.sin(theta) + Y * np.cos(theta)

    bar_half_width_px = deg_to_px(bar_width_deg / 2)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Current edge positions (symmetric expansion from center)
        edge_dist_deg = initial_separation_deg / 2 + edge_speed_deg_s * t
        edge_dist_deg = min(edge_dist_deg, max_expansion_deg)
        edge_dist_px = deg_to_px(edge_dist_deg)

        # Bar is visible between the two edges
        in_bar_length = np.abs(pos_along) <= edge_dist_px
        in_bar_width = np.abs(pos_perp) <= bar_half_width_px

        frame = create_blank_frame()

        # Only show the moving edges (thin lines at edge positions)
        edge_thickness_px = 3

        # Top edge (positive direction)
        top_edge = (np.abs(pos_along - edge_dist_px) < edge_thickness_px) & in_bar_width
        # Bottom edge (negative direction)
        bottom_edge = (np.abs(pos_along + edge_dist_px) < edge_thickness_px) & in_bar_width

        frame[top_edge | bottom_edge] = DARK if dark else BRIGHT

        frames[t_idx] = frame

    return frames


def generate_bar_expansion_filled(
    orientation_deg: float,
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 0.0,
    max_length_deg: float = 60.0,
    dark: bool = True
) -> np.ndarray:
    """
    Generate expanding bar stimulus with filled bar (not just edges).

    This version shows the entire bar expanding, which creates
    two dark edges moving outward from center.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    theta = np.radians(orientation_deg)
    pos_along = X * np.cos(theta) + Y * np.sin(theta)
    pos_perp = -X * np.sin(theta) + Y * np.cos(theta)

    bar_half_width_px = deg_to_px(bar_width_deg / 2)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Bar length grows with time
        bar_half_length_deg = initial_length_deg / 2 + edge_speed_deg_s * t
        bar_half_length_deg = min(bar_half_length_deg, max_length_deg / 2)
        bar_half_length_px = deg_to_px(bar_half_length_deg)

        # Bar mask
        in_length = np.abs(pos_along) <= bar_half_length_px
        in_width = np.abs(pos_perp) <= bar_half_width_px

        frame = create_blank_frame()
        frame[in_length & in_width] = DARK if dark else BRIGHT

        frames[t_idx] = frame

    return frames


# =============================================================================
# FIGURE 4A: CROSS MOTION STIMULI (DIAGONAL ORIENTATION)
# =============================================================================

def generate_disk_expansion(
    edge_speed_deg_s: float = 10.0,
    duration_s: float = 5.0,
    fps: int = FPS,
    initial_size_deg: float = 5.0,
    max_size_deg: float = 60.0,
    dark: bool = True
) -> np.ndarray:
    """
    Generate simple disk expansion stimulus.
    """
    return generate_constant_velocity_looming(
        edge_speed_deg_s=edge_speed_deg_s,
        duration_s=duration_s,
        fps=fps,
        initial_size_deg=initial_size_deg,
        max_size_deg=max_size_deg,
        dark=dark
    )


def generate_diagonal_cross_expansion(
    direction: str = 'outward',  # 'outward' or 'inward'
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 5.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0,
    include_center_square: bool = True,
    center_square_size_deg: float = 10.0
) -> np.ndarray:
    """
    Generate diagonal cross-shaped EXTENDING stimulus.

    Four bars along DIAGONAL axes (45°, 135°, 225°, 315°) that EXTEND
    from the center outward (or contract inward). The bars grow longer,
    they don't move as objects.

    Parameters:
    -----------
    direction : str
        'outward' for expansion (bars grow longer), 'inward' for contraction
    bar_width_deg : float
        Width of each bar
    edge_speed_deg_s : float
        Speed of bar extension
    initial_length_deg : float
        Initial length of each bar arm from center
    max_length_deg : float
        Maximum length of each bar arm from center
    include_center_square : bool
        Whether to include a static square at center
    center_square_size_deg : float
        Size of center square
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    bar_half_width_px = deg_to_px(bar_width_deg / 2)
    center_half_px = deg_to_px(center_square_size_deg / 2)

    # Diagonal angles (45°, 135°, 225°, 315°)
    diagonal_angles = [45, 135, 225, 315]

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Compute current bar length (extension from center)
        if direction == 'outward':
            current_length_deg = initial_length_deg + edge_speed_deg_s * t
        else:  # inward (contraction)
            current_length_deg = max_length_deg - edge_speed_deg_s * t
            current_length_deg = max(current_length_deg, initial_length_deg)

        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()

        all_bars = np.zeros_like(frame, dtype=bool)

        # Create four diagonal bars that EXTEND from center
        for angle_deg in diagonal_angles:
            theta = np.radians(angle_deg)

            # Rotate coordinate system to align with this diagonal
            # Position along the bar axis (from center outward)
            pos_along = X * np.cos(theta) + Y * np.sin(theta)
            # Position perpendicular to bar axis
            pos_perp = -X * np.sin(theta) + Y * np.cos(theta)

            # Bar extends from center (0) to current_length_px along positive direction
            # The bar is centered at the diagonal, with width bar_half_width_px
            in_length = (pos_along >= 0) & (pos_along <= current_length_px)
            in_width = np.abs(pos_perp) <= bar_half_width_px

            all_bars |= (in_length & in_width)

        # Center square (static) - rotated 45° to match diagonal cross
        if include_center_square:
            # Diamond-shaped center (rotated square)
            center_dist = np.abs(X) + np.abs(Y)  # Manhattan distance = diamond
            center_diamond = center_dist <= center_half_px * np.sqrt(2)
            all_bars |= center_diamond

        frame[all_bars] = DARK

        frames[t_idx] = frame

    return frames


def generate_cross_motion(
    direction: str = 'outward',  # 'outward' or 'inward'
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 5.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0,
    include_center_square: bool = True,
    center_square_size_deg: float = 10.0
) -> np.ndarray:
    """
    Wrapper for diagonal cross expansion (for backward compatibility).
    Now generates DIAGONAL extending cross, not cardinal moving bars.
    """
    return generate_diagonal_cross_expansion(
        direction=direction,
        bar_width_deg=bar_width_deg,
        edge_speed_deg_s=edge_speed_deg_s,
        duration_s=duration_s,
        fps=fps,
        initial_length_deg=initial_length_deg,
        max_length_deg=max_length_deg,
        include_center_square=include_center_square,
        center_square_size_deg=center_square_size_deg
    )


# =============================================================================
# FIGURE 4C: DIAGONAL AXIS DECOMPOSITION (EXTENDING BARS)
# =============================================================================

# Diagonal angles for reference
DIAGONAL_ANGLES = {
    'ne': 45,    # Northeast (up-right)
    'nw': 135,   # Northwest (up-left)
    'sw': 225,   # Southwest (down-left)
    'se': 315,   # Southeast (down-right)
}


def generate_diagonal_arm_extension(
    arms: List[str],  # List of 'ne', 'nw', 'sw', 'se' (diagonal directions)
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0,
    direction: str = 'outward'
) -> np.ndarray:
    """
    Generate EXTENDING bars along specific diagonal arms.

    Bars grow/extend from center outward (or contract inward).
    Uses diagonal orientations (45°, 135°, 225°, 315°).

    Parameters:
    -----------
    arms : List[str]
        Which diagonal arms to include ('ne', 'nw', 'sw', 'se')
        ne=45°, nw=135°, sw=225°, se=315°
    direction : str
        'outward' (bars grow longer) or 'inward' (bars shrink)
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()
    bar_half_width_px = deg_to_px(bar_width_deg / 2)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Compute current bar length
        if direction == 'outward':
            current_length_deg = initial_length_deg + edge_speed_deg_s * t
        else:  # inward
            current_length_deg = max_length_deg - edge_speed_deg_s * t
            current_length_deg = max(current_length_deg, initial_length_deg)

        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()
        all_bars = np.zeros_like(frame, dtype=bool)

        for arm in arms:
            if arm in DIAGONAL_ANGLES:
                theta = np.radians(DIAGONAL_ANGLES[arm])

                # Rotate coordinate system to align with this diagonal
                pos_along = X * np.cos(theta) + Y * np.sin(theta)
                pos_perp = -X * np.sin(theta) + Y * np.cos(theta)

                # Bar extends from center (0) to current_length_px
                in_length = (pos_along >= 0) & (pos_along <= current_length_px)
                in_width = np.abs(pos_perp) <= bar_half_width_px

                all_bars |= (in_length & in_width)

        frame[all_bars] = DARK
        frames[t_idx] = frame

    return frames


def generate_single_arm_motion(
    arms: List[str],  # Now uses diagonal: 'ne', 'nw', 'sw', 'se'
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0,
    direction: str = 'outward'
) -> np.ndarray:
    """
    Wrapper for diagonal arm extension (backward compatibility).
    Now uses diagonal extending bars.
    """
    return generate_diagonal_arm_extension(
        arms=arms,
        bar_width_deg=bar_width_deg,
        edge_speed_deg_s=edge_speed_deg_s,
        duration_s=duration_s,
        fps=fps,
        initial_length_deg=initial_length_deg,
        max_length_deg=max_length_deg,
        direction=direction
    )


def generate_outward_with_darkening(
    outward_arms: List[str],  # Diagonal arms with outward extension ('ne', 'nw', 'sw', 'se')
    darkening_arms: List[str],  # Diagonal arms with static darkening
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0
) -> np.ndarray:
    """
    Generate outward EXTENSION on some diagonal arms with static darkening on others.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()
    bar_half_width_px = deg_to_px(bar_width_deg / 2)

    for t_idx in range(n_frames):
        t = t_idx / fps
        t_norm = t / duration_s

        # Outward extension length
        current_length_deg = initial_length_deg + edge_speed_deg_s * t
        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()

        # Outward extending arms (dark)
        for arm in outward_arms:
            if arm in DIAGONAL_ANGLES:
                theta = np.radians(DIAGONAL_ANGLES[arm])
                pos_along = X * np.cos(theta) + Y * np.sin(theta)
                pos_perp = -X * np.sin(theta) + Y * np.cos(theta)

                in_length = (pos_along >= 0) & (pos_along <= current_length_px)
                in_width = np.abs(pos_perp) <= bar_half_width_px

                frame[in_length & in_width] = DARK

        # Darkening arms (static position, changing intensity)
        darkening_intensity = int(BACKGROUND - (BACKGROUND - DARK) * t_norm)
        darkening_length_px = deg_to_px(max_length_deg)

        for arm in darkening_arms:
            if arm in DIAGONAL_ANGLES:
                theta = np.radians(DIAGONAL_ANGLES[arm])
                pos_along = X * np.cos(theta) + Y * np.sin(theta)
                pos_perp = -X * np.sin(theta) + Y * np.cos(theta)

                # Static darkening region from near center to max length
                in_length = (pos_along >= deg_to_px(5)) & (pos_along <= darkening_length_px)
                in_width = np.abs(pos_perp) <= bar_half_width_px

                frame[in_length & in_width] = darkening_intensity

        frames[t_idx] = frame

    return frames


def generate_three_point_diagonal_motion(
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0,  # Must reach to center from circumference
    center_to_peripheral_deg: float = 30.0  # At circumference (RF radius)
) -> np.ndarray:
    """
    Generate stimulus with 3 origin points along a diagonal:
    - Center: expands in two directions (perpendicular to the diagonal axis)
    - Two peripheral points AT CIRCUMFERENCE: expand TOWARD the center (inward motion)

    This creates 2 outward motions (from center) + 2 inward motions (from periphery).
    End result should match 4C-d (full X shape).

    Parameters:
    -----------
    center_to_peripheral_deg : float
        Distance from center to each peripheral origin point (30° = at circumference)
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()
    bar_half_width_px = deg_to_px(bar_width_deg / 2)

    # The three points are along the NE-SW diagonal (45° / 225°)
    # Peripheral points at NE and SW positions
    diagonal_angle = 45  # degrees
    theta_diag = np.radians(diagonal_angle)

    # Perpendicular direction for center expansion (135° / 315°)
    theta_perp = np.radians(diagonal_angle + 90)

    peripheral_dist_px = deg_to_px(center_to_peripheral_deg)

    # Peripheral point positions
    ne_center_x = peripheral_dist_px * np.cos(theta_diag)
    ne_center_y = peripheral_dist_px * np.sin(theta_diag)
    sw_center_x = -peripheral_dist_px * np.cos(theta_diag)
    sw_center_y = -peripheral_dist_px * np.sin(theta_diag)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Current extension length
        current_length_deg = initial_length_deg + edge_speed_deg_s * t
        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()
        all_bars = np.zeros_like(frame, dtype=bool)

        # === CENTER: expands in perpendicular direction (NW and SE) ===
        # These are the two OUTWARD motions
        for sign in [1, -1]:  # Both directions perpendicular to diagonal
            # Position along perpendicular axis from center
            pos_along = sign * (X * np.cos(theta_perp) + Y * np.sin(theta_perp))
            pos_perp = np.abs(-X * np.sin(theta_perp) + Y * np.cos(theta_perp))

            in_length = (pos_along >= 0) & (pos_along <= current_length_px)
            in_width = pos_perp <= bar_half_width_px

            all_bars |= (in_length & in_width)

        # === PERIPHERAL POINTS: expand TOWARD center (inward) ===
        # NE point expands toward center (SW direction)
        X_ne = X - ne_center_x
        Y_ne = Y - ne_center_y
        # Direction toward center is SW (225°)
        pos_along_ne = -(X_ne * np.cos(theta_diag) + Y_ne * np.sin(theta_diag))
        pos_perp_ne = np.abs(-X_ne * np.sin(theta_diag) + Y_ne * np.cos(theta_diag))

        in_length_ne = (pos_along_ne >= 0) & (pos_along_ne <= current_length_px)
        in_width_ne = pos_perp_ne <= bar_half_width_px
        all_bars |= (in_length_ne & in_width_ne)

        # SW point expands toward center (NE direction)
        X_sw = X - sw_center_x
        Y_sw = Y - sw_center_y
        # Direction toward center is NE (45°)
        pos_along_sw = X_sw * np.cos(theta_diag) + Y_sw * np.sin(theta_diag)
        pos_perp_sw = np.abs(-X_sw * np.sin(theta_diag) + Y_sw * np.cos(theta_diag))

        in_length_sw = (pos_along_sw >= 0) & (pos_along_sw <= current_length_px)
        in_width_sw = pos_perp_sw <= bar_half_width_px
        all_bars |= (in_length_sw & in_width_sw)

        frame[all_bars] = DARK
        frames[t_idx] = frame

    return frames


# =============================================================================
# FIGURE 4E: BAR WIDTH EFFECTS (VERTICAL BARS)
# =============================================================================

def generate_vertical_bar_expansion(
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0
) -> np.ndarray:
    """
    Generate VERTICAL bar expansion (straight up and down from center).

    A bar centered horizontally that extends vertically in both directions.

    Parameters:
    -----------
    bar_width_deg : float
        Width of the bar (horizontal extent)
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()
    bar_half_width_px = deg_to_px(bar_width_deg / 2)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Current vertical extent
        current_length_deg = initial_length_deg + edge_speed_deg_s * t
        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()

        # Vertical bar: within width horizontally, extends vertically
        in_width = np.abs(X) <= bar_half_width_px
        in_length = np.abs(Y) <= current_length_px

        frame[in_width & in_length] = DARK

        frames[t_idx] = frame

    return frames


def generate_offaxis_bar_expansion(
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0
) -> np.ndarray:
    """
    Wrapper that now generates VERTICAL bar expansion.
    (Kept for backward compatibility)
    """
    return generate_vertical_bar_expansion(
        bar_width_deg=bar_width_deg,
        edge_speed_deg_s=edge_speed_deg_s,
        duration_s=duration_s,
        fps=fps,
        initial_length_deg=initial_length_deg,
        max_length_deg=max_length_deg
    )


# =============================================================================
# FIGURE 4G: CENTER BAR WITH SIDE EFFECTS
# =============================================================================

def generate_4g_center_bar(
    bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0
) -> np.ndarray:
    """
    Figure 4G-a: Same as 4E-10° - vertical bar expanding up/down from center.
    """
    return generate_vertical_bar_expansion(
        bar_width_deg=bar_width_deg,
        edge_speed_deg_s=edge_speed_deg_s,
        duration_s=duration_s,
        fps=fps,
        initial_length_deg=initial_length_deg,
        max_length_deg=max_length_deg
    )


def generate_4g_center_with_darkening_sides(
    center_bar_width_deg: float = 10.0,
    side_bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0,
    side_offset_deg: float = 25.0,  # Horizontal distance of side bars from center
    rf_radius_deg: float = 30.0
) -> np.ndarray:
    """
    Figure 4G-b: Center bar expanding up/down + left and right bars that DARKEN.
    Side bars span the diameter of the circle (static position, darkening intensity).
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    center_bar_half_width_px = deg_to_px(center_bar_width_deg / 2)
    side_bar_half_width_px = deg_to_px(side_bar_width_deg / 2)
    side_offset_px = deg_to_px(side_offset_deg)
    rf_radius_px = deg_to_px(rf_radius_deg)

    for t_idx in range(n_frames):
        t = t_idx / fps
        t_norm = t / duration_s

        # Center bar expansion
        current_length_deg = initial_length_deg + edge_speed_deg_s * t
        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()

        # Center bar (expanding up/down)
        center_in_width = np.abs(X) <= center_bar_half_width_px
        center_in_length = np.abs(Y) <= current_length_px
        frame[center_in_width & center_in_length] = DARK

        # Side bars (darkening) - span the full diameter vertically
        darkening_intensity = int(BACKGROUND - (BACKGROUND - DARK) * t_norm)

        # Left bar
        left_in_width = np.abs(X + side_offset_px) <= side_bar_half_width_px
        left_in_length = np.abs(Y) <= rf_radius_px
        frame[left_in_width & left_in_length] = darkening_intensity

        # Right bar
        right_in_width = np.abs(X - side_offset_px) <= side_bar_half_width_px
        right_in_length = np.abs(Y) <= rf_radius_px
        frame[right_in_width & right_in_length] = darkening_intensity

        frames[t_idx] = frame

    return frames


def generate_4g_center_with_expanding_sides(
    center_bar_width_deg: float = 10.0,
    side_bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0,
    side_offset_deg: float = 25.0  # Horizontal distance of side bars from center
) -> np.ndarray:
    """
    Figure 4G-c: Center bar expanding up/down + left and right bars that also EXPAND up/down.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    center_bar_half_width_px = deg_to_px(center_bar_width_deg / 2)
    side_bar_half_width_px = deg_to_px(side_bar_width_deg / 2)
    side_offset_px = deg_to_px(side_offset_deg)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Bar expansion (same for center and sides)
        current_length_deg = initial_length_deg + edge_speed_deg_s * t
        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()

        # Center bar (expanding up/down)
        center_in_width = np.abs(X) <= center_bar_half_width_px
        center_in_length = np.abs(Y) <= current_length_px
        frame[center_in_width & center_in_length] = DARK

        # Left bar (also expanding up/down)
        left_in_width = np.abs(X + side_offset_px) <= side_bar_half_width_px
        left_in_length = np.abs(Y) <= current_length_px
        frame[left_in_width & left_in_length] = DARK

        # Right bar (also expanding up/down)
        right_in_width = np.abs(X - side_offset_px) <= side_bar_half_width_px
        right_in_length = np.abs(Y) <= current_length_px
        frame[right_in_width & right_in_length] = DARK

        frames[t_idx] = frame

    return frames


def generate_4g_corners_inward(
        center_bar_width_deg: float = 10.0,
        side_bar_width_deg: float = 10.0,
        edge_speed_deg_s: float = 20.0,
        duration_s: float = 4.0,
        fps: int = FPS,
        initial_length_deg: float = 5.0,
        max_length_deg: float = 30.0,
        side_offset_deg: float = 25.0,
        rf_radius_deg: float = 30.0
) -> np.ndarray:
    """
    Figure 4G-d: Center bar expands UP/DOWN from center (outward motion),
    while side bars start at corners and expand INWARD toward center.

    - Center bar: expands from center outward (up and down)
    - Side bars: start at top/bottom corners, expand toward center (inward motion)
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    center_bar_half_width_px = deg_to_px(center_bar_width_deg / 2)
    side_bar_half_width_px = deg_to_px(side_bar_width_deg / 2)
    side_offset_px = deg_to_px(side_offset_deg)
    rf_radius_px = deg_to_px(rf_radius_deg)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Current expansion length
        current_length_deg = initial_length_deg + edge_speed_deg_s * t
        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()

        # === CENTER BAR: expands UP and DOWN from center (OUTWARD motion) ===
        center_in_width = np.abs(X) <= center_bar_half_width_px
        center_in_length = np.abs(Y) <= current_length_px
        frame[center_in_width & center_in_length] = DARK

        # === SIDE BARS: start at corners, expand INWARD toward center ===

        # Left bar
        left_in_width = np.abs(X + side_offset_px) <= side_bar_half_width_px

        # Top-left corner: starts at Y = -rf_radius, expands DOWN (toward center)
        top_left_in_length = (Y >= -rf_radius_px) & (Y <= -rf_radius_px + current_length_px)
        frame[left_in_width & top_left_in_length] = DARK

        # Bottom-left corner: starts at Y = +rf_radius, expands UP (toward center)
        bottom_left_in_length = (Y <= rf_radius_px) & (Y >= rf_radius_px - current_length_px)
        frame[left_in_width & bottom_left_in_length] = DARK

        # Right bar
        right_in_width = np.abs(X - side_offset_px) <= side_bar_half_width_px

        # Top-right corner: starts at Y = -rf_radius, expands DOWN (toward center)
        frame[right_in_width & top_left_in_length] = DARK

        # Bottom-right corner: starts at Y = +rf_radius, expands UP (toward center)
        frame[right_in_width & bottom_left_in_length] = DARK

        frames[t_idx] = frame

    return frames


def generate_4g_sides_only_expanding(
    side_bar_width_deg: float = 10.0,
    edge_speed_deg_s: float = 20.0,
    duration_s: float = 4.0,
    fps: int = FPS,
    initial_length_deg: float = 5.0,
    max_length_deg: float = 30.0,
    side_offset_deg: float = 25.0  # Horizontal distance of side bars from center
) -> np.ndarray:
    """
    Figure 4G-e: Same as 4G-c but WITHOUT the center bar.
    Only left and right bars expanding up/down from center.
    """
    n_frames = int(duration_s * fps)
    frames = np.zeros((n_frames, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

    X, Y = create_coordinate_grids()

    side_bar_half_width_px = deg_to_px(side_bar_width_deg / 2)
    side_offset_px = deg_to_px(side_offset_deg)

    for t_idx in range(n_frames):
        t = t_idx / fps

        # Bar expansion
        current_length_deg = initial_length_deg + edge_speed_deg_s * t
        current_length_deg = min(current_length_deg, max_length_deg)
        current_length_px = deg_to_px(current_length_deg)

        frame = create_blank_frame()

        # Left bar (expanding up/down from center)
        left_in_width = np.abs(X + side_offset_px) <= side_bar_half_width_px
        left_in_length = np.abs(Y) <= current_length_px
        frame[left_in_width & left_in_length] = DARK

        # Right bar (expanding up/down from center)
        right_in_width = np.abs(X - side_offset_px) <= side_bar_half_width_px
        right_in_length = np.abs(Y) <= current_length_px
        frame[right_in_width & right_in_length] = DARK

        # NO center bar

        frames[t_idx] = frame

    return frames


# Keep old functions for backward compatibility but mark as deprecated
def generate_center_motion(*args, **kwargs):
    """Deprecated: Use generate_4g_center_bar instead."""
    return generate_4g_center_bar(*args, **kwargs)

def generate_edge_motion(*args, **kwargs):
    """Deprecated: Old implementation removed."""
    return generate_4g_center_bar()  # Return something reasonable

def generate_center_and_edges(*args, **kwargs):
    """Deprecated: Use generate_4g_center_with_expanding_sides instead."""
    return generate_4g_center_with_expanding_sides(*args, **kwargs)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_stimulus(
    frames: np.ndarray,
    title: str = "Stimulus",
    fps: int = FPS,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a stimulus as a sequence of frames.

    Parameters:
    -----------
    frames : np.ndarray
        Array of shape (n_frames, height, width)
    title : str
        Title for the visualization
    fps : int
        Playback frame rate
    save_path : str, optional
        If provided, save frames as images to this directory
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=14)

    # Show sample frames at regular intervals
    n_samples = 8
    indices = np.linspace(0, len(frames) - 1, n_samples, dtype=int)

    for idx, (ax, frame_idx) in enumerate(zip(axes.flat, indices)):
        ax.imshow(frames[frame_idx], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"t = {frame_idx / fps:.2f}s")
        ax.axis('off')

        # Draw receptive field boundary
        circle = plt.Circle(
            (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
            deg_to_px(RF_RADIUS),
            fill=False,
            color='red',
            linestyle='--',
            linewidth=1
        )
        ax.add_patch(circle)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def create_stimulus_summary_figure(output_dir: str = '.') -> None:
    """
    Create a summary figure showing all stimulus types from the paper.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate all stimuli
    stimuli = {
        # Figure 2D
        'fig2d_dark_rv40': generate_looming_rv(rv_ms=40, dark=True),
        'fig2d_bright_rv40': generate_looming_rv(rv_ms=40, dark=False),
        'fig2d_receding': generate_looming_rv(rv_ms=40, is_receding=True),
        'fig2d_luminance': generate_luminance_control(),

        # Figure 2F
        'fig2f_looming': generate_constant_velocity_looming(edge_speed_deg_s=10),
        'fig2f_rightward': generate_widefield_translation(direction_deg=0),
        'fig2f_downward': generate_widefield_translation(direction_deg=90),
        'fig2f_rotation_cw': generate_rotation(direction=1),

        # Figure 3e/f
        'fig3e_bar10_0deg': generate_bar_expansion_filled(orientation_deg=0, bar_width_deg=10),
        'fig3e_bar10_45deg': generate_bar_expansion_filled(orientation_deg=45, bar_width_deg=10),
        'fig3f_bar60_0deg': generate_bar_expansion_filled(orientation_deg=0, bar_width_deg=60),
        'fig3f_bar60_45deg': generate_bar_expansion_filled(orientation_deg=45, bar_width_deg=60),

        # Figure 4A
        'fig4a_disk': generate_disk_expansion(),
        'fig4a_cross_out': generate_cross_motion(direction='outward'),
        'fig4a_cross_in': generate_cross_motion(direction='inward'),

        # Figure 4C
        'fig4c_single_arm': generate_single_arm_motion(arms=['up']),
        'fig4c_two_arms': generate_single_arm_motion(arms=['up', 'down']),
        'fig4c_full_axis': generate_single_arm_motion(arms=['up', 'down', 'left', 'right']),

        # Figure 4E
        'fig4e_bar10_45deg': generate_offaxis_bar_expansion(bar_width_deg=10),
        'fig4e_bar20_45deg': generate_offaxis_bar_expansion(bar_width_deg=20),
        'fig4e_bar60_45deg': generate_offaxis_bar_expansion(bar_width_deg=60),

        # Figure 4G
        'fig4g_center': generate_center_motion(),
        'fig4g_edges': generate_edge_motion(),
        'fig4g_center_edges': generate_center_and_edges(),
    }

    # Save visualizations
    for name, frames in stimuli.items():
        save_path = os.path.join(output_dir, f'{name}.png')
        visualize_stimulus(frames, title=name.replace('_', ' ').title(), save_path=save_path)

    print(f"\nAll stimulus visualizations saved to {output_dir}")
    return stimuli


# =============================================================================
# MAIN - GENERATE ALL VISUALIZATIONS
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("LPLC2 Visual Stimulus Generator")
    print("Klapoetke et al. 2017, Nature")
    print("=" * 60)

    # Create output directories
    viz_output_dir = '/mnt/user-data/outputs/lplc2_stimuli'
    npy_output_dir = './lplc2_inputs'  # Local folder for numpy arrays
    os.makedirs(viz_output_dir, exist_ok=True)
    os.makedirs(npy_output_dir, exist_ok=True)

    # Dictionary to store all stimuli
    all_stimuli = {}

    # Generate Figure 2D stimuli (Dark vs Bright looming at same r/v=40ms)
    print("\nGenerating Figure 2D stimuli (Dark vs Bright Looming at r/v=40ms)...")

    # Dark looming r/v=40ms
    frames = generate_looming_rv(rv_ms=40, dark=True)
    all_stimuli['fig2d_dark_rv40'] = frames
    np.save(f'{npy_output_dir}/fig2d_dark_rv40.npy', frames)
    visualize_stimulus(frames, "Fig 2D: Dark Looming r/v=40ms",
                      save_path=f'{viz_output_dir}/fig2d_dark_rv40.png')

    # Bright looming r/v=40ms
    frames = generate_looming_rv(rv_ms=40, dark=False)
    all_stimuli['fig2d_bright_rv40'] = frames
    np.save(f'{npy_output_dir}/fig2d_bright_rv40.npy', frames)
    visualize_stimulus(frames, "Fig 2D: Bright Looming r/v=40ms",
                      save_path=f'{viz_output_dir}/fig2d_bright_rv40.png')

    # Receding (dark)
    frames = generate_looming_rv(rv_ms=40, dark=True, is_receding=True)
    all_stimuli['fig2d_receding'] = frames
    np.save(f'{npy_output_dir}/fig2d_receding.npy', frames)
    visualize_stimulus(frames, "Fig 2D: Receding (Dark) r/v=40ms",
                      save_path=f'{viz_output_dir}/fig2d_receding.png')

    # Luminance control
    frames = generate_luminance_control()
    all_stimuli['fig2d_luminance'] = frames
    np.save(f'{npy_output_dir}/fig2d_luminance.npy', frames)
    visualize_stimulus(frames, "Fig 2D: Luminance Control",
                      save_path=f'{viz_output_dir}/fig2d_luminance.png')

    # Figure 2F - SKIPPED FOR NOW
    print("\nSkipping Figure 2F stimuli (wide-field motion)...")

    print("\nGenerating Figure 3e/f stimuli (Bar expansion)...")
    for width in [10, 60]:
        for orient in [0, 45, 90, 135]:
            frames = generate_bar_expansion_filled(orientation_deg=orient, bar_width_deg=width)
            key = f"fig3{'e' if width==10 else 'f'}_bar{width}_orient{orient}"
            all_stimuli[key] = frames
            np.save(f'{npy_output_dir}/{key}.npy', frames)
            visualize_stimulus(frames, f"Fig 3{'e' if width==10 else 'f'}: {width} deg bar, {orient} deg orientation",
                              save_path=f'{viz_output_dir}/fig3_bar{width}_orient{orient}.png')

    print("\nGenerating Figure 4A stimuli (Diagonal Cross Extension)...")

    # 4A-a: Disk expansion
    frames = generate_disk_expansion()
    all_stimuli['fig4a_disk'] = frames
    np.save(f'{npy_output_dir}/fig4a_disk.npy', frames)
    visualize_stimulus(frames, "Fig 4A-a: Disk Expansion",
                      save_path=f'{viz_output_dir}/fig4a_disk.png')

    # 4A-b: Diagonal cross EXTENDING outward
    frames = generate_diagonal_cross_expansion(direction='outward')
    all_stimuli['fig4a_cross_out'] = frames
    np.save(f'{npy_output_dir}/fig4a_cross_out.npy', frames)
    visualize_stimulus(frames, "Fig 4A-b: Diagonal Cross Extending Outward",
                      save_path=f'{viz_output_dir}/fig4a_cross_out.png')

    # 4A-c: Diagonal cross CONTRACTING inward
    frames = generate_diagonal_cross_expansion(direction='inward')
    all_stimuli['fig4a_cross_in'] = frames
    np.save(f'{npy_output_dir}/fig4a_cross_in.npy', frames)
    visualize_stimulus(frames, "Fig 4A-c: Diagonal Cross Contracting Inward",
                      save_path=f'{viz_output_dir}/fig4a_cross_in.png')

    print("\nGenerating Figure 4C stimuli (Diagonal axis decomposition - EXTENDING bars)...")

    # 4C-a: Single diagonal arm extending outward (NE direction)
    frames = generate_diagonal_arm_extension(arms=['ne'])
    all_stimuli['fig4c_single_ne'] = frames
    np.save(f'{npy_output_dir}/fig4c_single_ne.npy', frames)
    visualize_stimulus(frames, "Fig 4C-a: Single Arm Extending (NE)",
                      save_path=f'{viz_output_dir}/fig4c_single_ne.png')

    # 4C-b: Single arm in OPPOSITE direction (SW)
    frames = generate_diagonal_arm_extension(arms=['sw'])
    all_stimuli['fig4c_single_sw'] = frames
    np.save(f'{npy_output_dir}/fig4c_single_sw.npy', frames)
    visualize_stimulus(frames, "Fig 4C-b: Single Arm Extending (SW)",
                      save_path=f'{viz_output_dir}/fig4c_single_sw.png')

    # 4C-c: Full diagonal axis (two opposite arms) extending outward
    frames = generate_diagonal_arm_extension(arms=['ne', 'sw'])
    all_stimuli['fig4c_full_axis'] = frames
    np.save(f'{npy_output_dir}/fig4c_full_axis.npy', frames)
    visualize_stimulus(frames, "Fig 4C-c: Full Diagonal Axis Extending",
                      save_path=f'{viz_output_dir}/fig4c_full_axis.png')

    # 4C-d: One axis extending outward + orthogonal axis darkening
    frames = generate_outward_with_darkening(
        outward_arms=['ne', 'sw'],
        darkening_arms=['nw', 'se']
    )
    all_stimuli['fig4c_darkening'] = frames
    np.save(f'{npy_output_dir}/fig4c_darkening.npy', frames)
    visualize_stimulus(frames, "Fig 4C-d: Outward Extension + Darkening",
                      save_path=f'{viz_output_dir}/fig4c_darkening.png')

    # 4C-e: Three origin points - center expands perpendicular, periphery expands toward center
    frames = generate_three_point_diagonal_motion()
    all_stimuli['fig4c_three_point'] = frames
    np.save(f'{npy_output_dir}/fig4c_three_point.npy', frames)
    visualize_stimulus(frames, "Fig 4C-e: Three Points (2 outward + 2 inward)",
                      save_path=f'{viz_output_dir}/fig4c_three_point.png')

    print("\nGenerating Figure 4E stimuli (Bar width effects)...")
    for width in [10, 20, 60]:
        frames = generate_offaxis_bar_expansion(bar_width_deg=width)
        key = f'fig4e_bar{width}'
        all_stimuli[key] = frames
        np.save(f'{npy_output_dir}/{key}.npy', frames)
        visualize_stimulus(frames, f"Fig 4E: {width} deg Bar",
                          save_path=f'{viz_output_dir}/fig4e_bar{width}.png')

    print("\nGenerating Figure 4G stimuli (Center bar with side effects)...")

    # 4G-a: Same as 4E-10 - vertical bar expanding up/down
    frames = generate_4g_center_bar(bar_width_deg=10.0)
    all_stimuli['fig4g_center'] = frames
    np.save(f'{npy_output_dir}/fig4g_center.npy', frames)
    visualize_stimulus(frames, "Fig 4G-a: Center Bar Expanding (same as 4E-10)",
                      save_path=f'{viz_output_dir}/fig4g_center.png')

    # 4G-b: Center bar + left/right darkening bars
    frames = generate_4g_center_with_darkening_sides(center_bar_width_deg=10.0)
    all_stimuli['fig4g_darkening_sides'] = frames
    np.save(f'{npy_output_dir}/fig4g_darkening_sides.npy', frames)
    visualize_stimulus(frames, "Fig 4G-b: Center Bar + Side Darkening",
                      save_path=f'{viz_output_dir}/fig4g_darkening_sides.png')

    # 4G-c: Center bar + left/right expanding bars (all expand from center outward)
    frames = generate_4g_center_with_expanding_sides(center_bar_width_deg=10.0)
    all_stimuli['fig4g_expanding_sides'] = frames
    np.save(f'{npy_output_dir}/fig4g_expanding_sides.npy', frames)
    visualize_stimulus(frames, "Fig 4G-c: Center Bar + Side Expanding",
                      save_path=f'{viz_output_dir}/fig4g_expanding_sides.png')

    # 4G-d: Bars start at corners, expand INWARD (end = same as 4G-c)
    frames = generate_4g_corners_inward(side_bar_width_deg=10.0)
    all_stimuli['fig4g_corners_inward'] = frames
    np.save(f'{npy_output_dir}/fig4g_corners_inward.npy', frames)
    visualize_stimulus(frames, "Fig 4G-d: Corners Expanding Inward",
                      save_path=f'{viz_output_dir}/fig4g_corners_inward.png')

    # 4G-e: Same as 4G-c but WITHOUT center bar
    frames = generate_4g_sides_only_expanding(side_bar_width_deg=10.0)
    all_stimuli['fig4g_sides_only'] = frames
    np.save(f'{npy_output_dir}/fig4g_sides_only.npy', frames)
    visualize_stimulus(frames, "Fig 4G-e: Side Bars Only (no center)",
                      save_path=f'{viz_output_dir}/fig4g_sides_only.png')

    print("\n" + "=" * 60)
    print(f"Visualizations saved to: {viz_output_dir}")
    print(f"NumPy arrays saved to: {os.path.abspath(npy_output_dir)}")
    print(f"Total stimuli generated: {len(all_stimuli)}")
    print("\nTo load a stimulus in another program:")
    print("  import numpy as np")
    print("  frames = np.load('./lplc2_inputs/fig2d_dark_rv40.npy')")
    print("  # frames.shape = (n_frames, height, width), dtype=uint8")
    print("=" * 60)

    # At the end of your __main__ block:
    print("\nGenerating Figure 2F stimuli...")
    from fig2f_stimuli import generate_all_fig2f_stimuli

    fig2f_stimuli = generate_all_fig2f_stimuli(duration_s=5.0)
    for name, frames in fig2f_stimuli.items():
        np.save(f'{npy_output_dir}/{name}.npy', frames)
        all_stimuli[name] = frames
