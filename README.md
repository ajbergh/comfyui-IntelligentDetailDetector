# ComfyUI Intelligent Detail Detector

A custom node for ComfyUI that intelligently analyzes images to detect areas that would benefit from detail enhancement, outputting a weighted mask highlighting regions needing enhancement.

## Features

- **Intelligent Detail Detection**: Automatically identifies areas in an image that would benefit from additional detail or enhancement
- **Multi-method Analysis**: Combines edge detection, high-frequency content analysis, gradient magnitude, Laplacian, and focus measures to create a comprehensive detail map
- **Basic, Advanced, and Auto Modes**: Choose between simplified, full parameter control, or adaptive automatic parameter selection
- **Content-specific Presets**: Built-in presets optimized for different content types (Portrait, Landscape, Text/Document, Art/Drawing, etc.)
- **Visualization Options**: View the detail mask as a heatmap or overlay on the original image
- **Gradient Edge Softening**: Apply smooth transitions on mask edges
- **Facial Feature Detection**: Optional automatic face detection to emphasize facial details
- **Adaptive Threshold Calculation**: Automatically adjusts thresholds based on image statistics in Auto mode
- **Alpha Channel Support**: Properly handles RGBA images
- **Custom Parameter Override**: Use preset values as starting points and override individual parameters as needed
- **Fully Configurable**: Adjust sensitivity and weights to fine-tune detection for different image types
- **Outputs Both Image and Mask**: Returns both the original image and a weighted mask for use in other ComfyUI nodes

## Parameters

### Required Parameters
- **image**: Input image to analyze
- **preset**: Choose from predefined parameter sets (Default, Portrait, Landscape, Text/Document, Art/Drawing, Macro/Close-up, Low-light/Night)
- **mode**: Choose between "Basic", "Advanced", or "Auto" parameter sets

### Optional Parameters
- **detail_threshold** (0.1-1.0): Controls sensitivity for detail detection
- **edge_weight** (0-1.0): Adjusts importance of edges in the final mask
- **texture_weight** (0-1.0): Adjusts importance of texture details in the final mask
- **gradient_weight** (0-1.0): Controls influence of gradient magnitude detection
- **laplacian_weight** (0-1.0): Adjusts influence of Laplacian operator for edge detection
- **focus_weight** (0-1.0): Controls influence of focus/sharpness detection
- **blur_radius** (0-15): Controls smoothness of the output mask
- **visualize_mask**: Choose between "No" (original image), "Yes" (colorized heatmap), or "Overlay" (mask overlaid on image)
- **gradient_strength** (0-1.0): Adds gradient effect to soften mask edges
- **auto_threshold_strength** (0-1.0): Controls how strongly to apply automatic threshold adjustments in Auto mode
- **detect_faces**: Enable facial feature detection for enhanced detail in face regions
- **face_weight** (0-1.0): Controls the influence of detected facial features in the final mask
- **custom_preset**: Override preset values with manually specified parameters

## Presets

The node includes several presets optimized for different content types:

- **Default**: Balanced settings suitable for general images
- **Portrait**: Optimized for human faces with facial feature detection
- **Landscape**: Enhanced texture and edge detection for natural scenes
- **Text/Document**: Focused on high-contrast edges for text clarity
- **Art/Drawing**: Balanced settings for artistic works and illustrations
- **Macro/Close-up**: Emphasizes fine texture and detail detection
- **Low-light/Night**: Adjusted thresholds for low-contrast scenes

## Use Cases

- Pre-processing images for detail-enhancing upscalers
- Creating masks for selective detail enhancement in specific image regions
- Identifying areas that need more attention in image generation workflows
- Creating weighted control maps for ControlNet or similar modules
- Analyzing image quality and detail distribution
- Generating focus maps for depth estimation
- Facial detail preservation and enhancement in portraits
- Automatic adjustment for different lighting conditions and image types

## Integration

The node appears in the ComfyUI menu under the `image/analysis` category as "Intelligent Detail Detector".

## Basic Detail Enhancement Workflow

1. **Load Image Node** - Load your source image
   - Output: IMAGE

2. **Intelligent Detail Detector Node** - Generate detail mask
   - Inputs:
     - image: Connect to Load Image output
     - preset: "Default" (or choose based on image content)
     - mode: "Auto"
     - visualize_mask: "No" (to get original image and mask)
   - Outputs:
     - IMAGE: Original image
     - MASK: Detail mask

3. **Image Scale Node** - To upscale the original image
   - Inputs: 
     - image: Connect to Intelligent Detail Detector IMAGE output
     - width/height: Your target resolution
   - Output: Upscaled image

4. **Latent from Image Node** - Convert upscaled image to latents
   - Input: Connect to Image Scale output
   - Output: LATENT

5. **DetailerFix or ControlNet Node**
   - Inputs:
     - image: Connect to upscaled image
     - mask: Connect to Intelligent Detail Detector MASK output
     - model: Choose appropriate detail enhancer model
   - Output: Enhanced latent

6. **VAE Decode Node** - Convert back to image
   - Input: Connect to DetailerFix/ControlNet output
   - Output: Final enhanced image

## Advanced Detail-Targeted Workflow

1. **Load Image Node** → **Intelligent Detail Detector Node** (with "Yes" for visualize_mask to check detection)
   - Configure Intelligent Detail Detector with preset for your image type
   - Adjust parameters like detect_faces="Yes" for portraits

2. **Once detection is satisfactory**, change visualize_mask to "No" and:

3. **Image Scale Node** → Scale original image (from Intelligent Detail Detector output)

4. **Split workflow into two branches**:
   - Branch A: Apply standard upscale to base image
   - Branch B: 
     - Apply stronger detail enhancement using the mask
     - Apply stronger models only to detailed areas

5. **Mask Combine Node**:
   - Use the detail mask to blend results from branches A and B
   - Detailed areas get enhancement from Branch B
   - Non-detailed areas get standard treatment from Branch A

6. **Final Image Output**

## Face-Focused Enhancement Workflow

1. **Load Image Node** → **Intelligent Detail Detector Node**
   - Configure with:
     - preset: "Portrait"
     - detect_faces: "Yes"
     - face_weight: 0.9 (high emphasis on faces)

2. **A/B Compare with two branches**:
   - Branch A: Standard image upscaler
   - Branch B: Face-optimized detail enhancer using the detail mask

3. **Selective Enhancement**:
   - Apply standard enhancement to non-face areas
   - Apply face-optimized enhancement to face areas
   - Use the detail mask to control the blend

4. **Optional: Face Restoration Model** applied only to face regions using the mask

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/ajbergh/comfyui-IntelligentDetailDetector.git
```

2. Restart ComfyUI or reload the web interface

## License

[MIT License](LICENSE)