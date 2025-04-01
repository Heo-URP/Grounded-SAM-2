import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision.ops import box_convert

# Grounding DINO imports
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Diffusion imports
from diffusers import StableDiffusionInpaintPipeline

class GroundedSAMDiffusionPipeline:
    def __init__(self, 
                 sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
                 sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
                 grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth",
                 diffusion_model="stabilityai/stable-diffusion-2-inpainting",
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 box_threshold=0.35,
                 text_threshold=0.25):
        """
        Initialize the Grounded SAM + Diffusion pipeline for text-prompted object replacement.
        
        Args:
            sam2_checkpoint: Path to SAM2 checkpoint
            sam2_config: Path to SAM2 config file
            grounding_dino_config: Path to Grounding DINO config
            grounding_dino_checkpoint: Path to Grounding DINO checkpoint
            diffusion_model: Stable Diffusion model ID or path
            device: Device to run models on ('cuda' or 'cpu')
            box_threshold: Threshold for Grounding DINO box confidence
            text_threshold: Threshold for Grounding DINO text confidence
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        print(f"Using device: {device}")
        
        # Load SAM2 model
        print("Loading SAM2 model...")
        self.sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Load Grounding DINO model
        print("Loading Grounding DINO model...")
        self.grounding_model = load_model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
            device=device
        )
        
        # Load Stable Diffusion inpainting model
        print("Loading Stable Diffusion inpainting model...")
        self.diffusion = StableDiffusionInpaintPipeline.from_pretrained(
            diffusion_model,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
    
    def detect_and_segment(self, image_path, text_prompt):
        """
        Detect objects with Grounding DINO and segment them with SAM2.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for object detection (format: "object1. object2.")
            
        Returns:
            Original image, masks, input boxes, and class names
        """
        # Load image
        image_source, image = load_image(image_path)
        
        # Set image for SAM2 predictor
        self.sam2_predictor.set_image(image_source)
        
        # Detect objects with Grounding DINO
        boxes, confidences, class_names = predict(
            model=self.grounding_model,
            image=image,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        
        # Format boxes for SAM2
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # Generate masks with SAM2
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False
            )
        
        # Process masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        return image_source, masks, input_boxes, class_names, confidences.numpy()
    
    def replace_objects(self, image, masks, replacement_prompts):
        """
        Replace objects in the image using the diffusion model.
        
        Args:
            image: Original image (numpy array)
            masks: Segmentation masks from SAM2
            replacement_prompts: Prompts for replacement (can be a single string or list of strings)
            
        Returns:
            Edited image with objects replaced
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = image
            
        # Resize image to be compatible with diffusion model (multiple of 8)
        width, height = image_pil.size
        new_width = ((width + 7) // 8) * 8
        new_height = ((height + 7) // 8) * 8
        image_pil = image_pil.resize((new_width, new_height))
        
        # Process replacement prompts
        if isinstance(replacement_prompts, str):
            replacement_prompts = [replacement_prompts] * len(masks)
        elif len(replacement_prompts) != len(masks):
            replacement_prompts = replacement_prompts[:len(masks)] if len(replacement_prompts) > len(masks) else replacement_prompts + [replacement_prompts[-1]] * (len(masks) - len(replacement_prompts))
        
        # Process each mask and replace objects
        result_image = image_pil
        
        for i, mask in enumerate(masks):
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask.astype(np.uint8), (new_width, new_height))
            mask_pil = Image.fromarray(mask_resized * 255)
            
            # Run diffusion inpainting
            output = self.diffusion(
                prompt=replacement_prompts[i],
                image=result_image,
                mask_image=mask_pil,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]
            
            # Update result for next iteration
            result_image = output
        
        # Resize back to original dimensions if needed
        if (width, height) != (new_width, new_height):
            result_image = result_image.resize((width, height))
        
        return result_image
    
    def process_image(self, image_path, detection_prompt, replacement_prompts):
        """
        Process an image to detect, segment, and replace objects.
        
        Args:
            image_path: Path to input image
            detection_prompt: Text prompt for object detection (format: "object1. object2.")
            replacement_prompts: Prompts for object replacement
            
        Returns:
            Original image and result image with objects replaced
        """
        # Detect and segment objects
        image, masks, boxes, class_names, confidences = self.detect_and_segment(
            image_path=image_path,
            text_prompt=detection_prompt
        )
        
        if len(masks) == 0:
            print(f"No objects detected for prompt: {detection_prompt}")
            return image, Image.fromarray(image)
        
        # Print detected objects
        print(f"Detected {len(masks)} objects:")
        for i, (cls, conf) in enumerate(zip(class_names, confidences)):
            print(f"  {i+1}. {cls} (confidence: {conf:.2f})")
        
        # Replace objects
        result_image = self.replace_objects(
            image=image,
            masks=masks,
            replacement_prompts=replacement_prompts
        )
        
        return image, result_image
    
    def visualize_results(self, original_image, result_image, save_path=None):
        """
        Visualize original and result images side by side.
        
        Args:
            original_image: Original image (numpy array or PIL Image)
            result_image: Result image with objects replaced (PIL Image)
            save_path: Path to save visualization (optional)
        """
        # Convert to numpy if needed
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        if isinstance(original_image, np.ndarray) and original_image.shape[2] == 3 and original_image.dtype == np.uint8:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if original_image.shape[2] == 3 else original_image
        else:
            original_rgb = original_image
            
        # Display result
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result_image)
        plt.title("Replaced")
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()

# Example usage
if __name__ == "__main__":
    # Set your paths
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    grounding_dino_config = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_checkpoint = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    
    # Create pipeline
    pipeline = GroundedSAMDiffusionPipeline(
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config,
        grounding_dino_config=grounding_dino_config,
        grounding_dino_checkpoint=grounding_dino_checkpoint
    )
    
    # Process image
    input_image = "notebooks/images/cars.jpg"
    detection_prompt = "car."  # Note the period after each object
    replacement_prompts = ["blue bus", "yellow taxi"]

    original, result = pipeline.process_image(
        image_path=input_image, 
        detection_prompt=detection_prompt,
        replacement_prompts=replacement_prompts
    )
    
    filename = replacement_prompts[0].replace(" ", "_").replace("/", "_")

    # Visualize results
    pipeline.visualize_results(
        original, 
        result, 
        save_path=f"urp/{filename}.jpg"
    )