"""Legacy shim so `pip install -e .` works on boxes with an old setuptools that
lacks the PEP 660 build_editable hook. Metadata lives in pyproject.toml; this
just enables the src/ layout for editable installs everywhere."""
from setuptools import find_packages, setup

setup(
    name="abstention-under-constraint",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
