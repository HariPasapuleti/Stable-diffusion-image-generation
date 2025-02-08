# Stable Diffusion Image Generation

This repository contains a project for generating images using the Stable Diffusion model and Hugging Face's `diffusers` library. The project is implemented in Python and demonstrates the use of various machine learning tools and techniques for generating high-quality images from textual prompts.

## Features

- **Text-to-Image Generation:** Generates images based on textual descriptions using the Stable Diffusion model.
- **Configurable Parameters:** Easily adjust model settings like inference steps, guidance scale, and image size.
- **Hugging Face Integration:** Seamless integration with Hugging Face models and libraries.
- **CUDA Support:** Leverages GPU for faster inference.

## Demo

Watch the demo [here](https://www.youtube.com/watch?v=F6JGbFse62U).

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/HariPasapuleti/Stable-diffusion-image-generation
   cd stable-diffusion-image-generation
   ```

2. Install the required packages:
   ```bash
   pip install --upgrade diffusers transformers torch tqdm matplotlib opencv-python pandas numpy
   ```

3. (Optional) Log in to Hugging Face:
   ```python
   from huggingface_hub import login
   login("your-huggingface-token")
   ```

## Usage

1. Set up your configuration parameters such as device, seed, and model settings.
2. Load the Stable Diffusion pipeline from Hugging Face.
3. Provide a textual prompt, and the system will generate an image.

## Troubleshooting

- **Missing Files Error:** Ensure you have downloaded all required model files. Check the Hugging Face model page for completeness.
- **CUDA Errors:** Ensure your GPU drivers and `torch` version are compatible.
- **Authentication Warnings:** Log in to Hugging Face or set the `HF_AUTH_TOKEN` environment variable.

## Requirements

- Python 3.9+
- Libraries:
  - `diffusers`
  - `transformers`
  - `torch`
  - `tqdm`
  - `matplotlib`
  - `opencv-python`
  - `numpy`
- GPU

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing pre-trained models and libraries.
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for the foundational work on image generation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

