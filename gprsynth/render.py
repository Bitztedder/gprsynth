"""Array -> image. Grey by default; GPR is read as shape, not colour."""
import numpy as np
from PIL import Image


def to_image(sec, clip=99.0, invert=False):
    """sec (ntrace, nsample) -> PIL Image, rendered with the long axis horizontal."""
    a = np.asarray(sec, np.float64).T                    # (nsample, ntrace): depth downward
    lim = np.percentile(np.abs(a), clip) or 1.0
    a = np.clip(a / lim, -1, 1)
    g = ((a + 1) * 0.5 * 255).astype(np.uint8)
    if invert:
        g = 255 - g
    return Image.fromarray(g, mode="L")
