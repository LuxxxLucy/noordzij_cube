# Noordzij Cube VAE

Visualization of Gerrit Noordzij's typographic design space,
with a VAE for learning letter form latent space.

## Quick Start

This repo uses `uv`, the data set preprocessing can be found in `analysis/extract_letters.py` and `dataset/prepare_dataset.sh`

```bash
# Train the model
python train_vae.py
```

## References

- [Axis Praxis: Noordzij Cube](https://www.axis-praxis.org/playground/noordzij-cube/) The font source is from it.
- [In Memoriam Gerrit Noordzij 1931-2022](https://www.typeroom.eu/in-memoriam-gerrit-noordzij-1931-2022)
- [Multidimensional Axis Visualizer](https://fvar.unsoundscapes.com)
