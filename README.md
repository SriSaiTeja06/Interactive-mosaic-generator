# Interactive Image Mosaic Generator

Transform any image into an artistic mosaic using predefined colored tiles. This implementation demonstrates grid-based image processing with vectorized NumPy operations for optimal performance.

## Features

- **Grid-based Processing**: Configurable grid sizes from 16x16 to 64x64
- **Vectorized Operations**: NumPy broadcasting for 8.7x average speedup
- **Color Quantization**: Optional K-means clustering for enhanced quality
- **Real-time Processing**: Complete mosaic generation in under 0.03 seconds
- **Quality Metrics**: SSIM and MSE similarity assessment
- **Interactive Interface**: Gradio web interface with live preview

## Installation

```bash
# Clone the repository
git clone https://github.com/SriSaiTeja06/Interactive-mosaic-generator
cd interactive-mosaic-generator

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Dependencies

```
gradio==4.9.1
numpy==1.24.3
opencv-python==4.9.0.80
pillow==10.0.1
scikit-learn==1.3.0
scikit-image==0.21.0
```

## Usage

1. **Upload Image**: Select any image file (JPG, PNG, etc.)
2. **Configure Grid**: Adjust grid size for detail level
3. **Select Algorithm**: Choose vectorized or loop-based processing
4. **Apply Quantization**: Optional color reduction for artistic effects
5. **Generate Mosaic**: View original, segmented, and mosaic results

## Algorithm

The system processes images through four main stages:

1. **Preprocessing**: Resize to 400px max height, crop for grid alignment
2. **Grid Segmentation**: Divide image into cells, calculate average colors
3. **Tile Matching**: Find best-matching tiles using RGB distance
4. **Mosaic Assembly**: Replace each cell with corresponding tile

## Performance

Based on comprehensive testing with portrait imagery:

| Grid Size | Vectorized Time | Loop Time | SSIM Score | MSE Score |
|-----------|----------------|-----------|------------|-----------|
| 16x16     | 0.019s        | 0.029s    | 0.45-0.49  | 85-89     |
| 32x32     | 0.007s        | 0.070s    | 0.53-0.57  | 83-89     |
| 64x64     | 0.021s        | 0.276s    | 0.63-0.66  | 84-88     |

**Average Vectorized Speedup: 8.7x**

## Technical Details

- **Tile Library**: 111 predefined tiles across 10 color families
- **Color Matching**: Euclidean distance in RGB space
- **Grid Processing**: Vectorized NumPy array operations
- **Quality Assessment**: SSIM and MSE similarity metrics
- **Interface**: Professional Gradio web application

## File Structure

```
interactive-mosaic-generator/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── README.md          # Documentation
└── .gitignore         # Git exclusions
```

## Live Demo

Access the interactive demo: [https://huggingface.co/spaces/tej06/Mosaic_Generator]

## Results

The system successfully reconstructs recognizable images across all grid configurations:
- **Quality Range**: SSIM scores from 0.45 (blocky) to 0.66 (detailed)
- **Processing Speed**: Real-time generation under 0.03 seconds
- **Color Accuracy**: MSE scores consistently between 75-92
- **Scalability**: O(n²) complexity with excellent vectorized optimization

## Configuration Options

- **Grid Size**: Balance between detail and mosaic effect
- **Quantization**: Improves quality (+0.03-0.05 SSIM) at 20-25% time cost
- **Implementation**: Vectorized for speed, loops for educational comparison

## License

Open source project for educational purposes.

## Author

[Sri Sai Teja Mettu Srinivas]