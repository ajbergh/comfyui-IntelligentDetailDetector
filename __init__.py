import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import os

class IntelligentDetailDetector:
    """
    ComfyUI node that analyzes images to detect areas that would benefit from detail enhancement.
    Outputs a weighted mask highlighting regions needing enhancement.
    """
    
    # Define presets for different content types
    PRESETS = {
        "Default": {
            "detail_threshold": 0.5,
            "edge_weight": 0.7,
            "texture_weight": 0.6,
            "gradient_weight": 0.5,
            "laplacian_weight": 0.4,
            "focus_weight": 0.3,
            "blur_radius": 3,
            "gradient_strength": 0.0,
            "auto_threshold_strength": 0.5,
            "detect_faces": "No",
            "face_weight": 0.8,
        },
        "Portrait": {
            "detail_threshold": 0.4,
            "edge_weight": 0.8,
            "texture_weight": 0.7,
            "gradient_weight": 0.6,
            "laplacian_weight": 0.5,
            "focus_weight": 0.4,
            "blur_radius": 2,
            "gradient_strength": 0.2,
            "auto_threshold_strength": 0.6,
            "detect_faces": "Yes",
            "face_weight": 0.9,
        },
        "Landscape": {
            "detail_threshold": 0.6,
            "edge_weight": 0.8,
            "texture_weight": 0.9,
            "gradient_weight": 0.7,
            "laplacian_weight": 0.5,
            "focus_weight": 0.5,
            "blur_radius": 3,
            "gradient_strength": 0.4,
            "auto_threshold_strength": 0.7,
            "detect_faces": "No",
            "face_weight": 0.0,
        },
        "Text/Document": {
            "detail_threshold": 0.3,
            "edge_weight": 0.9,
            "texture_weight": 0.4,
            "gradient_weight": 0.6,
            "laplacian_weight": 0.7,
            "focus_weight": 0.3,
            "blur_radius": 1,
            "gradient_strength": 0.0,
            "auto_threshold_strength": 0.3,
            "detect_faces": "No",
            "face_weight": 0.0,
        },
        "Art/Drawing": {
            "detail_threshold": 0.45,
            "edge_weight": 0.85,
            "texture_weight": 0.7,
            "gradient_weight": 0.6,
            "laplacian_weight": 0.5,
            "focus_weight": 0.3,
            "blur_radius": 2,
            "gradient_strength": 0.2,
            "auto_threshold_strength": 0.5,
            "detect_faces": "No",
            "face_weight": 0.0,
        },
        "Macro/Close-up": {
            "detail_threshold": 0.35,
            "edge_weight": 0.8,
            "texture_weight": 0.9,
            "gradient_weight": 0.8,
            "laplacian_weight": 0.6,
            "focus_weight": 0.7,
            "blur_radius": 1,
            "gradient_strength": 0.1,
            "auto_threshold_strength": 0.6,
            "detect_faces": "No",
            "face_weight": 0.0,
        },
        "Low-light/Night": {
            "detail_threshold": 0.7,
            "edge_weight": 0.5,
            "texture_weight": 0.4,
            "gradient_weight": 0.4,
            "laplacian_weight": 0.3,
            "focus_weight": 0.2,
            "blur_radius": 5,
            "gradient_strength": 0.5,
            "auto_threshold_strength": 0.8,
            "detect_faces": "No",
            "face_weight": 0.5,
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (list(cls.PRESETS.keys()), {"default": "Default"}),
                "mode": (["Basic", "Advanced", "Auto"], {"default": "Basic"}),
            },
            "optional": {
                "detail_threshold": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.1, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "edge_weight": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "texture_weight": ("FLOAT", {
                    "default": 0.6, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "gradient_weight": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "laplacian_weight": ("FLOAT", {
                    "default": 0.4, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "focus_weight": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "blur_radius": ("INT", {
                    "default": 3, 
                    "min": 0, 
                    "max": 15, 
                    "step": 1
                }),
                "visualize_mask": (["No", "Yes", "Overlay"], {"default": "No"}),
                "gradient_strength": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "auto_threshold_strength": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "detect_faces": (["No", "Yes"], {"default": "No"}),
                "face_weight": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "custom_preset": (["No", "Yes"], {"default": "No"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "detect_detail_areas"
    CATEGORY = "image/analysis"

    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
    
    def load_face_detectors(self):
        """Load OpenCV face and eye detection cascades if not already loaded"""
        if self.face_cascade is None:
            # Try to find the cascade files in standard OpenCV locations
            opencv_path = cv2.__path__[0]
            data_path = os.path.join(opencv_path, 'data')
            
            # Default paths for cascade files
            face_cascade_path = os.path.join(data_path, 'haarcascade_frontalface_default.xml')
            eye_cascade_path = os.path.join(data_path, 'haarcascade_eye.xml')
            
            # Check if files exist, if not, use absolute paths
            if not os.path.exists(face_cascade_path):
                face_cascade_path = 'haarcascade_frontalface_default.xml'
            
            if not os.path.exists(eye_cascade_path):
                eye_cascade_path = 'haarcascade_eye.xml'
            
            try:
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
                self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            except Exception as e:
                print(f"Warning: Failed to load face detection cascades: {e}")
                self.face_cascade = None
                self.eye_cascade = None

    def detect_faces(self, img_gray, img_rgb):
        """
        Detect faces and create a mask highlighting facial features
        
        Args:
            img_gray: Grayscale image for detection
            img_rgb: Color image for visualization
            
        Returns:
            np.ndarray: Face mask with values 0-1
        """
        self.load_face_detectors()
        
        # Check if we have valid face detectors
        if self.face_cascade is None or self.eye_cascade is None:
            print("Face detection cascades not available")
            return np.zeros_like(img_gray, dtype=np.float32)
            
        # Create a mask for faces
        face_mask = np.zeros_like(img_gray, dtype=np.float32)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            img_gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # For each detected face, enhance the mask
        for (x, y, w, h) in faces:
            # Create a basic face region with higher weight
            face_roi = np.zeros_like(img_gray, dtype=np.float32)
            face_roi[y:y+h, x:x+w] = 0.7
            
            # Create a Gaussian weight distribution centered on the face
            # to give more emphasis to the center of the face
            y_indices, x_indices = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
            
            # Calculate center of the face
            face_center_y = y + h/2
            face_center_x = x + w/2
            
            # Create a 2D Gaussian centered at the face
            sigma = max(w, h) * 0.5  # Scale based on face size
            gaussian = np.exp(-((x_indices - face_center_x)**2 / (2 * sigma**2) + 
                               (y_indices - face_center_y)**2 / (2 * sigma**2)))
            
            # Add the Gaussian to the face mask, scaled to 0.3-1.0 range
            face_mask = np.maximum(face_mask, gaussian * 0.7 + 0.3)
            
            # Detect eyes within the face region for extra detail
            roi_gray = img_gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Add extra weight to eye regions
            for (ex, ey, ew, eh) in eyes:
                # Mark eye regions with even higher weight
                cv2.ellipse(face_mask, 
                           (x + ex + ew//2, y + ey + eh//2),
                           (ew//2 + 5, eh//2 + 5),  # Slightly larger than detected eye
                           0, 0, 360, 1.0, -1)  # Full weight for eyes
                           
            # Add extra weight to mouth region (approximated position)
            mouth_y = y + int(h * 0.7)  # Approximate mouth position at 70% of face height
            mouth_height = int(h * 0.2)  # Approximate mouth height
            cv2.rectangle(face_mask, 
                         (x + int(w * 0.25), mouth_y), 
                         (x + int(w * 0.75), mouth_y + mouth_height), 
                         0.9, -1)  # High weight for mouth region
        
        # Apply gaussian blur to smooth transitions
        if len(faces) > 0:
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
            
        return face_mask

    def calculate_adaptive_threshold(self, img_gray):
        """
        Calculate an adaptive threshold based on image statistics.
        
        Args:
            img_gray: Grayscale image array
            
        Returns:
            float: Adaptive threshold value between 0.1 and 0.9
        """
        # Calculate image statistics
        mean_val = np.mean(img_gray) / 255.0
        std_val = np.std(img_gray) / 255.0
        
        # Calculate entropy (measure of randomness/detail)
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        non_zero = hist > 0
        entropy = -np.sum(hist[non_zero] * np.log2(hist[non_zero]))
        norm_entropy = min(entropy / 8.0, 1.0)  # Normalize, 8 is max entropy for 8-bit image
        
        # Calculate gradient statistics
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(gradient_magnitude) / 255.0
        
        # Images with higher contrast, entropy, and gradient likely need lower thresholds
        # to detect more subtle details
        adaptive_threshold = 0.5 - (0.2 * std_val) - (0.15 * norm_entropy) - (0.15 * mean_gradient)
        
        # Ensure threshold stays in reasonable range
        return np.clip(adaptive_threshold, 0.1, 0.9)

    def detect_detail_areas(self, image, preset="Default", mode="Basic", 
                            detail_threshold=None, edge_weight=None, 
                            texture_weight=None, gradient_weight=None, 
                            laplacian_weight=None, focus_weight=None, 
                            blur_radius=None, visualize_mask="No", 
                            gradient_strength=None, auto_threshold_strength=None,
                            detect_faces=None, face_weight=None,
                            custom_preset="No"):
        """
        Analyze image to create a weighted mask of areas needing detail enhancement.
        
        Args:
            image: Input image tensor from ComfyUI (B,H,W,C)
            preset: Preset to use for parameter values
            mode: Processing mode (Basic, Advanced, or Auto)
            detail_threshold: Sensitivity for detail detection
            edge_weight: Importance of edges in the final mask
            texture_weight: Importance of textures in the final mask
            gradient_weight: Importance of gradient in the final mask
            laplacian_weight: Importance of laplacian in the final mask
            focus_weight: Importance of focus measure in the final mask
            blur_radius: Radius for mask smoothing
            visualize_mask: How to return the mask (No, Yes, Overlay)
            gradient_strength: Strength of gradient effect to soften mask edges
            auto_threshold_strength: How strongly to apply the auto threshold adjustment
            detect_faces: Whether to apply face detection enhancement
            face_weight: Weight to apply to detected facial features
            custom_preset: Whether to use provided values instead of preset values
            
        Returns:
            tuple: (original_image or visualization, detail_mask)
        """
        # Apply preset parameters if custom_preset is not enabled
        if custom_preset == "No" and preset in self.PRESETS:
            preset_values = self.PRESETS[preset]
            # Only use preset values if the parameter is None or at its default value
            # This allows for partial override of preset values
            detail_threshold = preset_values["detail_threshold"] if detail_threshold is None else detail_threshold
            edge_weight = preset_values["edge_weight"] if edge_weight is None else edge_weight
            texture_weight = preset_values["texture_weight"] if texture_weight is None else texture_weight
            gradient_weight = preset_values["gradient_weight"] if gradient_weight is None else gradient_weight
            laplacian_weight = preset_values["laplacian_weight"] if laplacian_weight is None else laplacian_weight
            focus_weight = preset_values["focus_weight"] if focus_weight is None else focus_weight
            blur_radius = preset_values["blur_radius"] if blur_radius is None else blur_radius
            gradient_strength = preset_values["gradient_strength"] if gradient_strength is None else gradient_strength
            auto_threshold_strength = preset_values["auto_threshold_strength"] if auto_threshold_strength is None else auto_threshold_strength
            detect_faces = preset_values["detect_faces"] if detect_faces is None else detect_faces
            face_weight = preset_values["face_weight"] if face_weight is None else face_weight
        else:
            # Use provided values or defaults
            detail_threshold = 0.5 if detail_threshold is None else detail_threshold
            edge_weight = 0.7 if edge_weight is None else edge_weight
            texture_weight = 0.6 if texture_weight is None else texture_weight
            gradient_weight = 0.5 if gradient_weight is None else gradient_weight
            laplacian_weight = 0.4 if laplacian_weight is None else laplacian_weight
            focus_weight = 0.3 if focus_weight is None else focus_weight
            blur_radius = 3 if blur_radius is None else blur_radius
            gradient_strength = 0.0 if gradient_strength is None else gradient_strength
            auto_threshold_strength = 0.5 if auto_threshold_strength is None else auto_threshold_strength
            detect_faces = "No" if detect_faces is None else detect_faces
            face_weight = 0.8 if face_weight is None else face_weight
            
        # Convert from ComfyUI tensor format to OpenCV format
        # ComfyUI uses BHWC format with values 0-1
        img_np = (255. * image[0].cpu().numpy()).astype(np.uint8)
        
        # Convert to grayscale for analysis
        if img_np.shape[2] == 4:  # RGBA
            img_gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
            img_rgb = img_np[:, :, :3]
            alpha = img_np[:, :, 3]
        else:  # RGB
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_rgb = img_np
            alpha = None
            
        # Apply smart threshold if in Auto mode
        if mode == "Auto":
            adaptive_threshold = self.calculate_adaptive_threshold(img_gray)
            # Blend between user threshold and adaptive threshold based on strength
            detail_threshold = detail_threshold * (1 - auto_threshold_strength) + adaptive_threshold * auto_threshold_strength
        
        # Create mask based on multiple detail detection methods
        mask = np.zeros_like(img_gray, dtype=np.float32)
        
        # 1. Edge detection (captures boundaries where details matter)
        edges = cv2.Canny(img_gray, 100, 200)
        # Dilate edges to create areas of interest around them
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        mask += edge_weight * (edges_dilated / 255.0)
        
        # 2. High-frequency content detection (captures textures and fine details)
        # Apply blur and subtract from original to find high-frequency content
        blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)
        high_freq = cv2.absdiff(img_gray, blurred)
        # Normalize and threshold to focus on significant high-frequency areas
        high_freq_norm = high_freq / high_freq.max() if high_freq.max() > 0 else high_freq
        _, high_freq_thresh = cv2.threshold(
            high_freq_norm, detail_threshold, 1.0, cv2.THRESH_BINARY
        )
        mask += texture_weight * high_freq_thresh
        
        # 3. Gradient magnitude (captures areas with significant changes)
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_normalized = gradient_magnitude / gradient_magnitude.max() if gradient_magnitude.max() > 0 else gradient_magnitude
        mask += gradient_weight * gradient_normalized
        
        # 4. Add Laplacian for additional detail detection
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        laplacian_normalized = np.abs(laplacian) / np.max(np.abs(laplacian)) if np.max(np.abs(laplacian)) > 0 else np.abs(laplacian)
        mask += laplacian_weight * laplacian_normalized
        
        # 5. Focus measure - variance of Laplacian (for sharpness detection)
        def variance_of_laplacian(image):
            lap = cv2.Laplacian(image, cv2.CV_64F)
            lap_var = np.var(lap)
            # Create a normalized map based on local variance
            kernel_size = 15
            local_lap_var = cv2.GaussianBlur(lap * lap, (kernel_size, kernel_size), 0)
            local_lap_var = np.sqrt(local_lap_var)
            return local_lap_var / local_lap_var.max() if local_lap_var.max() > 0 else local_lap_var
            
        focus_map = variance_of_laplacian(img_gray)
        mask += focus_weight * focus_map
        
        # 6. Add face detection enhancement if enabled
        if detect_faces == "Yes":
            face_mask = self.detect_faces(img_gray, img_rgb)
            # Blend face mask with existing mask using face_weight
            if np.max(face_mask) > 0:  # Only if faces were detected
                mask = mask * (1 - face_weight) + face_mask * face_weight
        
        # Normalize the combined mask to 0-1 range
        mask = np.clip(mask, 0, 1)
        
        # Apply optional smoothing to the mask
        if blur_radius > 0:
            mask = cv2.GaussianBlur(mask, (blur_radius*2+1, blur_radius*2+1), 0)
        
        # Add a gradient effect to soften the mask edges if requested
        if gradient_strength > 0:
            # Create distance map from mask edges
            _, binary_mask = cv2.threshold(mask, 0.2, 1.0, cv2.THRESH_BINARY)
            dist_transform = cv2.distanceTransform((binary_mask * 255).astype(np.uint8), cv2.DIST_L2, 3)
            dist_transform_inv = cv2.distanceTransform(((1-binary_mask) * 255).astype(np.uint8), cv2.DIST_L2, 3)
            
            # Normalize distance transforms
            max_dist = max(np.max(dist_transform), np.max(dist_transform_inv))
            if max_dist > 0:
                dist_transform /= max_dist
                dist_transform_inv /= max_dist
            
            # Apply gradient effect
            gradient_factor = np.minimum(dist_transform, dist_transform_inv) 
            gradient_factor = np.clip(gradient_factor / (gradient_strength * 10 + 1e-5), 0, 1)
            mask = mask * (1 - gradient_strength) + mask * gradient_factor * gradient_strength
        
        # Convert the mask back to ComfyUI tensor format
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
        
        # Handle visualization options
        if visualize_mask == "Yes":
            # Create a heatmap visualization of the mask
            heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0) / 255.0
            return (heatmap_tensor, mask_tensor)
        elif visualize_mask == "Overlay":
            # Create an overlay of the mask on the original image
            heatmap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Convert original image back to RGB if needed
            if img_np.shape[2] == 4:
                img_rgb = img_np[:, :, :3]
            else:
                img_rgb = img_np
                
            # Blend the heatmap with the original image
            overlay = cv2.addWeighted(img_rgb, 0.7, heatmap, 0.3, 0)
            overlay_tensor = torch.from_numpy(overlay).unsqueeze(0) / 255.0
            return (overlay_tensor, mask_tensor)
        else:
            # Return original behavior
            return (image, mask_tensor)

# Node class registration function for ComfyUI
NODE_CLASS_MAPPINGS = {
    "IntelligentDetailDetector": IntelligentDetailDetector
}

# Optional: Add human-readable name display for UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "IntelligentDetailDetector": "Intelligent Detail Detector"
}