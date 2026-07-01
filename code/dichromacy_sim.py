"""
dichromacy_sim.py

Wraps DichromRenderLinear into a single convenience function that accepts
a standard (H, W, 3) sRGB image and returns the dichromat-simulated sRGB image.

Intended usage in the training pipeline:
    from dichromacy_sim import DichromatSimulateLinear, load_display
    Disp = load_display()                           # call once
    di_img = DichromatSimulateLinear(img, 'Deuteranopia', Disp)

Run this file directly to verify the simulator with a before/after plot:
    python dichromacy_sim.py
"""

import sys
import numpy as np

# Add display/ and helpers/ to sys.path so loadDisplay and DichromRenderLinear
# can be imported regardless of where this file is run from
DISPLAY_PATH = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/display'
HELPERS_PATH = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/Denoiser_Reconstruction/helpers'
for _p in [DISPLAY_PATH, HELPERS_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loadDisplay import loadDisplay
from DichromRenderLinear import DichromRenderLinear


def load_display():
    """Load display parameters (call once, then pass Disp into DichromatSimulateLinear)."""
    return loadDisplay()


def _srgb_to_linear(img):
    # Undo sRGB gamma correction (IEC 61966-2-1) to get physical linear light
    return np.where(img <= 0.04045,
                    img / 12.92,
                    ((img + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(img):
    # Re-apply sRGB gamma so the result displays correctly on a monitor
    img = np.clip(img, 0, 1)
    return np.where(img <= 0.0031308,
                    img * 12.92,
                    1.055 * img ** (1 / 2.4) - 0.055)


def DichromatSimulateLinear(rgb_image, dichromat_type, Disp=None):
    """
    Simulate how a dichromat sees an sRGB image.

    Parameters
    ----------
    rgb_image : np.ndarray, shape (H, W, 3), float32 in [0, 1], sRGB
    dichromat_type : str
        One of: 'Deuteranopia' (missing M cone, red/green confusion)
                'Protanopia'   (missing L cone, red/green confusion)
                'Tritanopia'   (missing S cone, blue/yellow confusion)
    Disp : Disp object or None
        Output of load_display(). If None, loadDisplay() is called automatically
        (slow -- better to call load_display() once and reuse Disp).

    Returns
    -------
    di_rgb : np.ndarray, shape (H, W, 3), float32 in [0, 1], sRGB
        The dichromat-simulated image, ready to display or save.
    """
    if Disp is None:
        Disp = load_display()

    H, W, _ = rgb_image.shape

    # Step 1: undo sRGB gamma -> linear light values (physical intensities)
    img_linear = _srgb_to_linear(rgb_image.astype(np.float64))

    # Step 2: reshape to cal format (3, H*W) -- one column per pixel
    rgb_cal = img_linear.reshape(-1, 3).T          # (3, H*W)

    # Step 3: linear RGB -> LMS cone excitations using the monitor's
    # M_rgb2cones matrix (3x3, derived from T_cones @ P_monitor)
    lms_cal = Disp.M_rgb2cones @ rgb_cal           # (3, H*W)

    # Step 4: simulate dichromacy.
    # DichromRenderLinear projects LMS cone contrast onto the 2D plane
    # spanned by the achromatic direction and a monochromatic constraint,
    # which fills in the missing cone value from the two available cones.
    # It returns calFormatDirgbLin: already converted back to linear RGB.
    _, rgb_lin_di_cal, _ = DichromRenderLinear(lms_cal, dichromat_type, Disp)

    # Step 5: reshape back to (H, W, 3) and re-apply sRGB gamma
    rgb_lin_di = rgb_lin_di_cal.T.reshape(H, W, 3)
    di_rgb = _linear_to_srgb(rgb_lin_di).astype(np.float32)

    return di_rgb


# -----------------------------------------------------------------------
# Run this block when the file is executed directly (python dichromacy_sim.py)
# to verify the simulator with a before/after image
# -----------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image

    Disp = load_display()

    IMG_PATH = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/Denoiser_Reconstruction/flower1.png'
    img_srgb = np.array(Image.open(IMG_PATH).convert('RGB')).astype(np.float32) / 255.0
    print(f'Loaded: {IMG_PATH}  shape: {img_srgb.shape}')

    types = ['Deuteranopia', 'Protanopia', 'Tritanopia']
    simulated = {t: DichromatSimulateLinear(img_srgb, t, Disp) for t in types}

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(img_srgb)
    axs[0].set_title('Original (trichromat)', fontsize=11)
    axs[0].axis('off')

    short_labels = {
        'Deuteranopia': 'Deuteranopia\n(missing M cone)',
        'Protanopia':   'Protanopia\n(missing L cone)',
        'Tritanopia':   'Tritanopia\n(missing S cone)',
    }
    for ax, t in zip(axs[1:], types):
        ax.imshow(simulated[t])
        ax.set_title(short_labels[t], fontsize=10)
        ax.axis('off')

    fig.suptitle('DichromatSimulateLinear — before/after verification', fontsize=13)
    fig.tight_layout()

    OUT = '/Users/callista/Documents/MATLAB/projects/ColorCorrectionRecon/texture_model/test_dichromacy_output.png'
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    print(f'Saved to: {OUT}')
    plt.show()
