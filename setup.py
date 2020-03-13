"""Setup.py for gdec package."""
from setuptools import setup

setup(
    name="gdec",
    version="0.0.1",
    author="C. Daniel Greenidge",
    author_email="dev@danielgreenidge.com",
    description="Linear decoders for angled grating stimuli",
    packages=["gdec"],
    install_requires=["numpy", "scikit-learn", "scipy"],
)
