# Reproduction environment

- Python 3.12.12
- twpasolver 0.0.1
- NumPy 1.26.4
- Matplotlib 3.11.1
- scikit-rf 2.0.1
- Numba 0.66.0
- Pydantic 2.13.4
- CyRK 0.8.8
- nbformat 5.10.4
- nbclient 0.11.0
- Platform used for initial validation: macOS 14.5, Apple silicon

On this machine the prebuilt CyRK extensions required an additional runtime
search path for `/opt/homebrew/opt/libomp/lib`. Reinstalling CyRK may overwrite
that local binary repair.

