"""CoDARec: Co-evolutionary Diffusion-Augmented Recommendation.

A PyTorch implementation of a denoising-diffusion recommender that couples a
generative diffusion model with a PPO-based denoising policy.
"""

from codarec.config import Config

__version__ = "1.0.0"
__all__ = ["Config", "__version__"]
