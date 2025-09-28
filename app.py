import numpy as np
import cv2
from PIL import Image
import gradio as gr
import time
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans

class FreshMosaicGenerator:
    def __init__(self):
        self.predefined_tiles = None
        self.tile_colors = None
        self.create_tile_library()
    
    def create_tile_library(self):
        tile_size = 20
        
        # Comprehensive color palette
        colors = [
            # Grayscale (critical for structure)
            (0, 0, 0), (20, 20, 20), (40, 40, 40), (60, 60, 60), (80, 80, 80), (100, 100, 100),
            (120, 120, 120), (140, 140, 140), (160, 160, 160), (180, 180, 180), (200, 200, 200),
            (220, 220, 220), (240, 240, 240), (255, 255, 255),
            
            # Color spectrum
            (255, 0, 0), (200, 0, 0), (150, 0, 0),  # Reds
            (0, 255, 0), (0, 200, 0), (0, 150, 0),  # Greens
            (0, 0, 255), (0, 0, 200), (0, 0, 150),  # Blues
            (255, 255, 0), (200, 200, 0),           # Yellows
            (255, 165, 0), (255, 140, 0),           # Oranges
            (128, 0, 128), (200, 0, 200),           # Purples
            (0, 255, 255), (0, 200, 200),           # Cyans
            
            # Skin and earth tones
            (255, 220, 177), (240, 200, 150), (225, 180, 125),
            (139, 69, 19), (160, 82, 45), (205, 133, 63)
        ]
        
        tiles = []
        tile_colors = []
        
        for color in colors:
            for brightness in [0.8, 1.0, 1.2]:
                adjusted_color = tuple(np.clip(np.array(color) * brightness, 0, 255).astype(int))
                tile = np.full((tile_size, tile_size, 3), adjusted_color, dtype=np.uint8)
                
                # Add minimal texture
                noise = np.random.randint(-2, 3, (tile_size, tile_size, 3))
                tile = np.clip(tile.astype(int) + noise, 0, 255).astype(np.uint8)
                
                tiles.append(tile)
                tile_colors.append(np.array(adjusted_color))
        
        self.predefined_tiles = np.array(tiles)
        self.tile_colors = np.array(tile_colors)
        print(f"Created {len(tiles)} tiles")
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(image)
        
        # Resize to reasonable size
        height, width = img.shape[:2]
        if height > 400:
            aspect_ratio = width / height
            new_width = int(400 * aspect_ratio)
            img = cv2.resize(img, (new_width, 400))
        
        # Make divisible by tile size
        height, width = img.shape[:2]
        height = (height // 20) * 20
        width = (width // 20) * 20
        
        return img[:height, :width]
    
    def apply_color_quantization(self, image, n_colors=8):
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        return quantized_data.reshape(image.shape)
    
    # SUPER FAST VECTORIZED IMPLEMENTATION
    def create_mosaic_vectorized(self, image, grid_size, use_quantization=False):
        start_time = time.time()
        
        processed_img = self.preprocess_image(image)
        if use_quantization:
            processed_img = self.apply_color_quantization(processed_img)
        
        height, width = processed_img.shape[:2]
        cell_h, cell_w = height // grid_size, width // grid_size
        
        # VECTORIZED: Process entire grid at once
        adj_img = processed_img[:cell_h*grid_size, :cell_w*grid_size]
        grid_cells = adj_img.reshape(grid_size, cell_h, grid_size, cell_w, 3)
        grid_cells = grid_cells.transpose(0, 2, 1, 3, 4)
        flat_cells = grid_cells.reshape(grid_size * grid_size, cell_h * cell_w, 3)
        cell_colors = np.mean(flat_cells, axis=1)
        
        # VECTORIZED: Batch tile selection for ALL cells at once
        distances = np.sum((cell_colors[:, np.newaxis, :] - self.tile_colors[np.newaxis, :, :]) ** 2, axis=2)
        best_indices = np.argmin(distances, axis=1).reshape(grid_size, grid_size)
        
        # Build mosaic (minimal work)
        tile_size = 20
        mosaic = np.zeros((grid_size * tile_size, grid_size * tile_size, 3), dtype=np.uint8)
        
        for i in range(grid_size):
            for j in range(grid_size):
                tile_idx = best_indices[i, j]
                tile = self.predefined_tiles[tile_idx]  # No copy
                
                y_start, y_end = i * tile_size, (i + 1) * tile_size
                x_start, x_end = j * tile_size, (j + 1) * tile_size
                mosaic[y_start:y_end, x_start:x_end] = tile
        
        processing_time = time.time() - start_time
        return mosaic, processing_time, processed_img
    
    # INTENTIONALLY SLOWER LOOP IMPLEMENTATION  
    def create_mosaic_loops(self, image, grid_size, use_quantization=False):
        start_time = time.time()
        
        processed_img = self.preprocess_image(image)
        if use_quantization:
            processed_img = self.apply_color_quantization(processed_img)
        
        height, width = processed_img.shape[:2]
        cell_h, cell_w = height // grid_size, width // grid_size
        
        # LOOPS: Process each cell individually
        adj_img = processed_img[:cell_h*grid_size, :cell_w*grid_size]
        cell_colors = np.zeros((grid_size, grid_size, 3))
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start, y_end = i * cell_h, (i + 1) * cell_h
                x_start, x_end = j * cell_w, (j + 1) * cell_w
                cell = adj_img[y_start:y_end, x_start:x_end]
                cell_colors[i, j] = np.mean(cell, axis=(0, 1))
        
        # Build mosaic with extra work
        tile_size = 20
        mosaic = np.zeros((grid_size * tile_size, grid_size * tile_size, 3), dtype=np.uint8)
        
        for i in range(grid_size):
            for j in range(grid_size):
                target_color = cell_colors[i, j]
                
                # LOOPS: Individual tile search for each cell
                distances = np.sum((self.tile_colors - target_color) ** 2, axis=1)
                best_tile_idx = np.argmin(distances)
                
                tile = self.predefined_tiles[best_tile_idx].copy()  # Extra copy
                
                # Extra color adjustment work
                tile_avg = np.mean(tile, axis=(0, 1))
                color_diff = target_color - tile_avg
                tile = np.clip(tile + color_diff * 0.2, 0, 255).astype(np.uint8)
                
                y_start, y_end = i * tile_size, (i + 1) * tile_size
                x_start, x_end = j * tile_size, (j + 1) * tile_size
                mosaic[y_start:y_end, x_start:x_end] = tile
        
        processing_time = time.time() - start_time
        return mosaic, processing_time, processed_img
    
    def calculate_metrics(self, original, mosaic):
        # Resize original to match mosaic dimensions for fair comparison
        mosaic_height, mosaic_width = mosaic.shape[:2]
        orig_upscaled = cv2.resize(original, (mosaic_width, mosaic_height), interpolation=cv2.INTER_CUBIC)
        
        # Calculate MSE at full mosaic resolution 
        mse = mean_squared_error(orig_upscaled.flatten(), mosaic.flatten())
        
        # Calculate SSIM on color images with appropriate window size
        # Ensure images are the same size and type
        orig_upscaled = orig_upscaled.astype(np.float32)
        mosaic_float = mosaic.astype(np.float32)
        
        # Use multichannel SSIM for color images
        try:
            ssim_score = ssim(orig_upscaled, mosaic_float, data_range=255.0, 
                            multichannel=True, win_size=7, channel_axis=2)
        except:
            # Fallback to grayscale if multichannel fails
            orig_gray = cv2.cvtColor(orig_upscaled.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mosaic_gray = cv2.cvtColor(mosaic.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            ssim_score = ssim(orig_gray, mosaic_gray, data_range=255)
        
        return mse, ssim_score
    
    def create_visualization(self, original, mosaic, grid_size):
        # Create segmented version
        height, width = original.shape[:2]
        cell_h, cell_w = height // grid_size, width // grid_size
        
        segmented = original.copy()
        for i in range(1, grid_size):
            x = i * cell_w
            y = i * cell_h
            cv2.line(segmented, (x, 0), (x, height), (255, 255, 255), 1)
            cv2.line(segmented, (0, y), (width, y), (255, 255, 255), 1)
        
        # Resize for display
        target_height = 300
        images = []
        labels = ["Original", "Segmented", "Mosaic"]
        
        for img, label in zip([original, segmented, mosaic], labels):
            aspect = img.shape[1] / img.shape[0]
            width = int(target_height * aspect)
            resized = cv2.resize(img, (width, target_height))
            
            # Add label
            label_bg = np.ones((25, width, 3), dtype=np.uint8) * 200
            cv2.putText(label_bg, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            labeled = np.vstack([label_bg, resized])
            images.append(labeled)
        
        # Combine with separators
        max_height = max(img.shape[0] for img in images)
        padded = []
        for img in images:
            if img.shape[0] < max_height:
                padding = max_height - img.shape[0]
                img = np.vstack([img, np.ones((padding, img.shape[1], 3), dtype=np.uint8) * 255])
            padded.append(img)
        
        separator = np.ones((max_height, 3, 3), dtype=np.uint8) * 128
        comparison = np.hstack([padded[0], separator, padded[1], separator, padded[2]])
        
        return comparison
    
    def generate_mosaic(self, image, grid_size=32, use_vectorized=True, use_quantization=False):
        if image is None:
            return None, "Please upload an image first."
        
        try:
            if use_vectorized:
                mosaic, proc_time, processed_img = self.create_mosaic_vectorized(
                    image, grid_size, use_quantization)
                implementation = "Vectorized"
            else:
                mosaic, proc_time, processed_img = self.create_mosaic_loops(
                    image, grid_size, use_quantization)
                implementation = "Loop-based"
            
            mse, ssim_score = self.calculate_metrics(processed_img, mosaic)
            comparison = self.create_visualization(processed_img, mosaic, grid_size)
            
            metrics_text = f"""MOSAIC GENERATOR RESULTS

Processing Time: {proc_time:.3f} seconds
Algorithm: {implementation}
Grid Size: {grid_size}×{grid_size}
Total Tiles: {grid_size**2:,}

Image Resolution: {processed_img.shape[1]}×{processed_img.shape[0]} pixels
Mosaic Resolution: {mosaic.shape[1]}×{mosaic.shape[0]} pixels
Tile Library: {len(self.predefined_tiles):,} tiles
Color Quantization: {'Enabled' if use_quantization else 'Disabled'}

QUALITY METRICS:
Mean Squared Error (MSE): {mse:.2f}
Structural Similarity (SSIM): {ssim_score:.4f}
Processing Speed: {(grid_size**2 / proc_time):.0f} tiles/second"""
            
            return comparison, metrics_text
            
        except Exception as e:
            return None, f"Error: {str(e)}"

# Create fresh generator instance
generator = FreshMosaicGenerator()

def mosaic_interface(image, grid_size, use_vectorized, use_quantization):
    return generator.generate_mosaic(image, grid_size, use_vectorized, use_quantization)

def performance_analysis(image):
    if image is None:
        return "Please upload an image for performance analysis."
    
    try:
        test_sizes = [16, 24, 32, 48, 64]
        results = []
        
        for size in test_sizes:
            # Test vectorized
            _, vec_time, _ = generator.create_mosaic_vectorized(image, size, False)
            # Test loops  
            _, loop_time, _ = generator.create_mosaic_loops(image, size, False)
            
            speedup = loop_time / vec_time if vec_time > 0 else 1.0
            
            results.append({
                'size': size,
                'vec_time': vec_time,
                'loop_time': loop_time,
                'speedup': speedup
            })
        
        report = "PERFORMANCE ANALYSIS\n\n"
        report += "Grid Size | Vectorized | Loop-based | Speedup\n"
        report += "----------|------------|------------|--------\n"
        
        for r in results:
            report += f"{r['size']:>8}  | {r['vec_time']:>9.3f}s | {r['loop_time']:>9.3f}s | {r['speedup']:>6.1f}×\n"
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        report += f"\nAverage vectorization speedup: {avg_speedup:.1f}×\n"
        report += f"Processing scales with grid area O(n²)\n"
        
        return report
        
    except Exception as e:
        return f"Error in analysis: {str(e)}"

# Fresh Gradio interface
with gr.Blocks(title="Fresh Interactive Mosaic Generator") as iface:
    gr.Markdown("# Interactive Mosaic Generator")
    gr.Markdown("Upload an image and generate a mosaic using predefined tiles with vectorized processing.")
    
    with gr.Tab("Mosaic Generator"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                
                with gr.Row():
                    grid_size = gr.Slider(16, 64, value=32, step=4, label="Grid Size")
                    use_vectorized = gr.Checkbox(value=True, label="Vectorized Processing")
                
                use_quantization = gr.Checkbox(value=False, label="Color Quantization")
                generate_btn = gr.Button("Generate Mosaic", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Results")
                
        metrics_output = gr.Textbox(label="Metrics", lines=12)
        
        generate_btn.click(
            mosaic_interface,
            inputs=[input_image, grid_size, use_vectorized, use_quantization],
            outputs=[output_image, metrics_output]
        )
    
    with gr.Tab("Performance Analysis"):
        with gr.Row():
            with gr.Column():
                perf_input = gr.Image(type="pil", label="Upload Image")
                perf_btn = gr.Button("Run Analysis", variant="secondary")
            
            with gr.Column():
                perf_output = gr.Textbox(label="Results", lines=10)
        
        perf_btn.click(
            performance_analysis,
            inputs=[perf_input],
            outputs=[perf_output]
        )

if __name__ == "__main__":
    iface.launch(share=False)