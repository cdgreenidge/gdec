"""Setup.py for gdec package."""
from setuptools import setup

setup(
    name="gdec",
    version="0.0.9",
    author="C. Daniel Greenidge",
    author_email="dev@danielgreenidge.com",
    description="Linear decoders for angled grating stimuli",
    packages=["gdec"],
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "scikit-learn",
        "scipy",
        "tqdm",
        "torch",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "flake8-bugbear",
            "flake8-blind-except",
            "flake8-builtins",
            "flake8-docstrings",
            "flake8-mutable",
            "hypothesis",
            "isort",
            "mypy",
            "pep8-naming",
            "pytest",
        ]
    },
)
