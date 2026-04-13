
# LLaVA 1.5 Vision-Language Alignment Dashboard

A [visual demonstration](https://llava-alignment.streamlit.app) of how MLPs bridge the gap between visual and textual representations in modern multimodal models.

## Overview

This project analyzes the **LLaVA 1.5** model to understand how its MLP projector aligns vision and language embeddings. It generates interactive visualizations showing:

- **Cosine Similarity Distributions**: Compares embedding similarities before (raw CLIP patches) and after MLP projection
- **t-SNE Projections**: Visualizes how text and image patches cluster in embedding space

## Components

### `generate_assets.py`
Processes sample images and text through LLaVA 1.5 to:
- Download images from URLs
- Extract raw and projected visual embeddings
- Compute alignment metrics
- Generate distribution and t-SNE plots

### `app.py`
A Streamlit dashboard displaying:
- Input images (optimized for desktop and mobile)
- Cosine similarity distributions before/after projection
- Interactive t-SNE visualization

## Key Insight

The MLP projector transforms raw CLIP visual embeddings to better align with text embeddings, improving multimodal understanding in vision-language models.
