"""
3D Mesh Generation Module using Hunyuan3D

This module handles 3D mesh generation from inpainted object images using
Hunyuan3D-Omni for geometry generation and Hunyuan3D-2.1 for texture painting.

References:
- Hunyuan3D-Omni: https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni
- Hunyuan3D-2.1: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
"""
import os
import numpy as np
import torch
import trimesh
from typing import List, Optional, Literal
from pathlib import Path
import time
from PIL import Image

from ..core.common import DetectedObject, Mesh3D, DepthEstimation
from ..utils.image_utils import save_image
from ..config.settings import config


class HunyuanMeshGenerationModule:
    """Module for 3D mesh generation using Hunyuan3D pipeline"""
    
    def __init__(self, 
                 omni_model_path: Optional[str] = None,
                 texture_model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize Hunyuan3D mesh generation module
        
        Args:
            omni_model_path: Path to Hunyuan3D-Omni model weights
            texture_model_path: Path to Hunyuan3D-2.1 texture model weights
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.omni_model_path = omni_model_path
        self.texture_model_path = texture_model_path
        
        # Models will be loaded lazily
        self.omni_model = None
        self.texture_model = None
        
        print(f"Initialized HunyuanMeshGenerationModule on device: {self.device}")
    
    def load_omni_model(self):
        """Load Hunyuan3D-Omni model for geometry generation"""
        if self.omni_model is not None:
            return
        
        try:
            print("Loading Hunyuan3D-Omni model...")
            
            # Import Hunyuan3D-Omni dependencies
            try:
                from hunyuan3d_omni.inference import load_model, generate_geometry
                self.generate_geometry_fn = generate_geometry
            except ImportError:
                raise ImportError(
                    "Hunyuan3D-Omni not found. Please install it:\n"
                    "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni.git\n"
                    "cd Hunyuan3D-Omni && pip install -e ."
                )
            
            # Load the model
            model_path = self.omni_model_path or "tencent/Hunyuan3D-Omni"
            self.omni_model = load_model(model_path, device=self.device)
            
            print("Hunyuan3D-Omni model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Hunyuan3D-Omni model: {e}")
            raise
    
    def load_texture_model(self):
        """Load Hunyuan3D-2.1 texture painting model"""
        if self.texture_model is not None:
            return
        
        try:
            print("Loading Hunyuan3D-2.1 texture model...")
            
            # Import Hunyuan3D-2.1 dependencies
            try:
                from hunyuan3d_2.inference import load_texture_model, paint_texture
                self.paint_texture_fn = paint_texture
            except ImportError:
                raise ImportError(
                    "Hunyuan3D-2.1 not found. Please install it:\n"
                    "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git\n"
                    "cd Hunyuan3D-2.1 && pip install -e ."
                )
            
            # Load the texture model
            model_path = self.texture_model_path or "tencent/Hunyuan3D-2.1"
            self.texture_model = load_texture_model(model_path, device=self.device)
            
            print("Hunyuan3D-2.1 texture model loaded successfully")
            
        except Exception as e:
            print(f"Error loading Hunyuan3D-2.1 texture model: {e}")
            raise
    
    def prepare_point_cloud_condition(self, 
                                     detected_object: DetectedObject,
                                     depth_estimation: DepthEstimation) -> np.ndarray:
        """
        Extract and prepare point cloud for the detected object as conditional input
        
        Args:
            detected_object: Object with mask information
            depth_estimation: Scene depth estimation with point cloud
            
        Returns:
            Prepared point cloud as numpy array (N, 3)
        """
        if detected_object.mask is None:
            raise ValueError(f"No mask available for object {detected_object.id}")
        
        # Get the full scene point cloud
        points_3d = depth_estimation.point_cloud  # (H, W, 3)
        
        # Apply object mask to extract object-specific points
        mask = detected_object.mask
        if mask.dtype != bool:
            mask = mask > 127
        
        # Extract points within the object mask
        object_points = points_3d[mask]
        
        # Remove invalid points (zeros or NaN)
        valid_indices = ~np.any(np.isnan(object_points) | (object_points == 0), axis=1)
        object_points = object_points[valid_indices]
        
        # Convert to OpenGL coordinates (consistent with existing pipeline)
        # OpenCV: +X right, +Y down, +Z forward
        # OpenGL: +X right, +Y up, +Z backward
        opengl_points = object_points.copy()
        opengl_points[:, 1] = -opengl_points[:, 1]  # Flip Y
        opengl_points[:, 2] = -opengl_points[:, 2]  # Flip Z
        
        print(f"Extracted {len(opengl_points)} points for object {detected_object.id}")
        
        return opengl_points
    
    def generate_geometry_with_pointcloud(self,
                                         image: np.ndarray,
                                         point_cloud: np.ndarray,
                                         output_path: Path) -> Optional[Path]:
        """
        Generate 3D geometry using Hunyuan3D-Omni with point cloud conditioning
        
        Args:
            image: Input RGB image (inpainted object)
            point_cloud: Conditional point cloud (N, 3)
            output_path: Path to save the generated geometry
            
        Returns:
            Path to generated geometry file (GLB format)
        """
        # Load model if not already loaded
        self.load_omni_model()
        
        print(f"Generating geometry with Hunyuan3D-Omni...")
        print(f"  Input image shape: {image.shape}")
        print(f"  Point cloud shape: {point_cloud.shape}")
        
        try:
            # Convert image to PIL format
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            
            # Prepare point cloud tensor
            point_cloud_tensor = torch.from_numpy(point_cloud).float().to(self.device)
            
            # Run Hunyuan3D-Omni inference
            # Following the interface from Hunyuan3D-Omni inference.py
            with torch.no_grad():
                geometry_output = self.generate_geometry_fn(
                    model=self.omni_model,
                    image=pil_image,
                    point_cloud=point_cloud_tensor,
                    device=self.device
                )
            
            # Save geometry to output path
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            # The output should contain mesh vertices and faces
            if isinstance(geometry_output, dict):
                vertices = geometry_output.get('vertices')
                faces = geometry_output.get('faces')
                
                if vertices is not None and faces is not None:
                    # Convert to numpy if tensors
                    if torch.is_tensor(vertices):
                        vertices = vertices.cpu().numpy()
                    if torch.is_tensor(faces):
                        faces = faces.cpu().numpy()
                    
                    # Create and save mesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    mesh.export(str(output_path))
                    
                    print(f"  Generated geometry saved to: {output_path}")
                    return output_path
            
            print("  Error: Invalid geometry output format")
            return None
            
        except Exception as e:
            print(f"  Error generating geometry: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def paint_texture_on_geometry(self,
                                 geometry_path: Path,
                                 input_image: np.ndarray,
                                 output_path: Path) -> Optional[Path]:
        """
        Paint texture on geometry using Hunyuan3D-2.1
        
        Args:
            geometry_path: Path to generated geometry (GLB)
            input_image: Original RGB image for texture reference
            output_path: Path to save the textured mesh
            
        Returns:
            Path to textured mesh file
        """
        # Load texture model if not already loaded
        self.load_texture_model()
        
        print(f"Painting texture with Hunyuan3D-2.1...")
        print(f"  Geometry: {geometry_path}")
        print(f"  Input image shape: {input_image.shape}")
        
        try:
            # Convert image to PIL format
            if input_image.dtype == np.float32 or input_image.dtype == np.float64:
                input_image = (input_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(input_image)
            
            # Load geometry mesh
            geometry_mesh = trimesh.load(str(geometry_path))
            if isinstance(geometry_mesh, trimesh.Scene):
                # Extract first mesh from scene
                geometry_mesh = list(geometry_mesh.geometry.values())[0]
            
            # Run texture painting inference
            # Following the interface from Hunyuan3D-2.1 demo.py
            with torch.no_grad():
                textured_mesh = self.paint_texture_fn(
                    model=self.texture_model,
                    mesh=geometry_mesh,
                    image=pil_image,
                    device=self.device
                )
            
            # Save textured mesh
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
            if textured_mesh is not None:
                # Export textured mesh
                if isinstance(textured_mesh, trimesh.Trimesh):
                    textured_mesh.export(str(output_path))
                elif isinstance(textured_mesh, dict) and 'mesh' in textured_mesh:
                    # Handle case where output is a dict
                    mesh_to_save = textured_mesh['mesh']
                    if isinstance(mesh_to_save, trimesh.Trimesh):
                        mesh_to_save.export(str(output_path))
                    else:
                        print("  Error: Invalid mesh format in output")
                        return None
                
                print(f"  Textured mesh saved to: {output_path}")
                return output_path
            else:
                print("  Error: Texture painting returned None")
                return None
            
        except Exception as e:
            print(f"  Error painting texture: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_mesh_for_object(self,
                                detected_object: DetectedObject,
                                depth_estimation: DepthEstimation,
                                output_dir: Optional[Path] = None) -> Optional[Mesh3D]:
        """
        Generate 3D mesh for a single object using Hunyuan3D pipeline
        
        This method performs:
        1. Geometry generation using Hunyuan3D-Omni (with point cloud conditioning)
        2. Texture painting using Hunyuan3D-2.1
        
        Args:
            detected_object: Object with inpainted image
            depth_estimation: Scene depth estimation for point cloud extraction
            output_dir: Directory to save the mesh file
            
        Returns:
            Mesh3D object if successful, None otherwise
        """
        if detected_object.generated_image is None and detected_object.cropped_image is None:
            print(f"Warning: No inpainted image for object {detected_object.id}")
            return None
        
        print(f"Generating 3D mesh with Hunyuan3D for object {detected_object.id}: {detected_object.description}")
        
        try:
            # Use generated image if available, otherwise use cropped image
            input_image = detected_object.generated_image if detected_object.generated_image is not None else detected_object.cropped_image
            
            # Step 1: Extract point cloud for this object
            print("  Step 1: Extracting point cloud...")
            point_cloud = self.prepare_point_cloud_condition(detected_object, depth_estimation)
            
            if len(point_cloud) < 100:
                print(f"  Warning: Too few points ({len(point_cloud)}) for object {detected_object.id}")
                # Continue anyway, the model might handle sparse input
            
            # Step 2: Generate geometry with Hunyuan3D-Omni
            print("  Step 2: Generating geometry with Hunyuan3D-Omni...")
            if output_dir:
                mesh_dir = output_dir / "meshes"
                mesh_dir.mkdir(exist_ok=True, parents=True)
                geometry_path = mesh_dir / f"object_{detected_object.id}_geometry.glb"
            else:
                geometry_path = Path(f"object_{detected_object.id}_geometry.glb")
            
            geometry_path = self.generate_geometry_with_pointcloud(
                input_image,
                point_cloud,
                geometry_path
            )
            
            if geometry_path is None or not geometry_path.exists():
                print(f"  Failed to generate geometry for object {detected_object.id}")
                return None
            
            # Step 3: Paint texture with Hunyuan3D-2.1
            print("  Step 3: Painting texture with Hunyuan3D-2.1...")
            if output_dir:
                textured_path = mesh_dir / f"object_{detected_object.id}_hunyuan.glb"
            else:
                textured_path = Path(f"object_{detected_object.id}_hunyuan.glb")
            
            textured_path = self.paint_texture_on_geometry(
                geometry_path,
                input_image,
                textured_path
            )
            
            if textured_path is None or not textured_path.exists():
                print(f"  Warning: Texture painting failed, using geometry-only mesh")
                textured_path = geometry_path
            
            # Step 4: Load and return the final mesh
            mesh_3d = self._load_mesh_from_file(textured_path)
            if mesh_3d:
                mesh_3d.input_image = input_image
                print(f"  Successfully generated mesh for object {detected_object.id}")
                return mesh_3d
            else:
                print(f"  Failed to load generated mesh for object {detected_object.id}")
                return None
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error generating mesh for object {detected_object.id}: {e}")
            return None
    
    def _load_mesh_from_file(self, mesh_path: Path) -> Optional[Mesh3D]:
        """
        Load mesh from file using trimesh
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Mesh3D object if successful
        """
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(str(mesh_path))
            
            # Handle different mesh types
            if isinstance(mesh, trimesh.Scene):
                # If it's a scene, get the first mesh
                mesh_geometries = [geom for geom in mesh.geometry.values() 
                                 if isinstance(geom, trimesh.Trimesh)]
                if mesh_geometries:
                    mesh = mesh_geometries[0]
                else:
                    print(f"No valid mesh found in scene file: {mesh_path}")
                    return None
            
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Loaded object is not a valid mesh: {mesh_path}")
                return None
            
            # Extract vertices and faces
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            
            # Extract texture/color information if available
            textures = None
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                textures = np.array(mesh.visual.vertex_colors)
            elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                textures = np.array(mesh.visual.face_colors)
            
            # Create Mesh3D object
            mesh_3d = Mesh3D(
                vertices=vertices,
                faces=faces,
                textures=textures,
                file_path=mesh_path
            )
            
            print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
            return mesh_3d
            
        except Exception as e:
            print(f"Error loading mesh from {mesh_path}: {e}")
            return None
    
    def batch_generate_meshes(self,
                             detected_objects: List[DetectedObject],
                             depth_estimation: DepthEstimation,
                             output_dir: Optional[Path] = None) -> List[Optional[Mesh3D]]:
        """
        Generate meshes for multiple objects sequentially
        
        Args:
            detected_objects: List of objects to generate meshes for
            depth_estimation: Scene depth estimation
            output_dir: Directory to save mesh files
            
        Returns:
            List of Mesh3D objects (None for failed generations)
        """
        print(f"Starting batch mesh generation for {len(detected_objects)} objects using Hunyuan3D...")
        
        # Filter objects with valid images
        valid_objects = [obj for obj in detected_objects 
                        if obj.generated_image is not None or obj.cropped_image is not None]
        
        if not valid_objects:
            print("No valid objects with inpainted images for Hunyuan3D generation")
            return []
        
        # Generate meshes sequentially (Hunyuan3D models are GPU-intensive)
        meshes = []
        for i, obj in enumerate(valid_objects):
            print(f"Processing object {i+1}/{len(valid_objects)}...")
            mesh = self.generate_mesh_for_object(obj, depth_estimation, output_dir)
            meshes.append(mesh)
        
        successful_meshes = sum(1 for mesh in meshes if mesh is not None)
        print(f"Batch mesh generation complete. {successful_meshes}/{len(valid_objects)} successful.")
        
        return meshes
    
    def run(self,
            detected_objects: List[DetectedObject],
            depth_estimation: DepthEstimation,
            output_dir: Optional[Path] = None) -> List[Optional[Mesh3D]]:
        """
        Run the complete Hunyuan3D mesh generation pipeline
        
        Args:
            detected_objects: List of objects with inpainted images
            depth_estimation: Scene depth estimation for point cloud extraction
            output_dir: Directory to save results
            
        Returns:
            List of generated meshes
        """
        print("Starting Hunyuan3D mesh generation pipeline...")
        
        if not detected_objects:
            print("No objects with inpainted images found")
            return []
        
        print(f"Generating meshes for {len(detected_objects)} objects...")
        
        # Generate meshes
        meshes = self.batch_generate_meshes(detected_objects, depth_estimation, output_dir)
        
        # Filter out None values and pair with objects
        valid_meshes = [mesh for mesh in meshes if mesh is not None]
        
        if len(valid_meshes) != len(detected_objects):
            print(f"Warning: Some mesh generations failed. {len(valid_meshes)}/{len(detected_objects)} successful.")
        
        # Save summary
        if output_dir and valid_meshes:
            self._save_summary(detected_objects, meshes, output_dir)
        
        print("Hunyuan3D mesh generation pipeline complete.")
        return meshes
    
    def _save_summary(self,
                     detected_objects: List[DetectedObject],
                     meshes: List[Optional[Mesh3D]],
                     output_dir: Path) -> None:
        """Save mesh generation summary"""
        mesh_dir = output_dir / "meshes"
        mesh_dir.mkdir(exist_ok=True, parents=True)
        
        summary = {
            "provider": "hunyuan3d",
            "total_objects": len(detected_objects),
            "successful_meshes": sum(1 for mesh in meshes if mesh is not None),
            "failed_meshes": sum(1 for mesh in meshes if mesh is None),
            "objects": []
        }
        
        for i, (obj, mesh) in enumerate(zip(detected_objects, meshes)):
            if mesh is not None and mesh.input_image is not None:
                save_image(mesh.input_image, mesh_dir / f"object_{obj.id}_input_image.png")
            
            obj_summary = {
                "id": obj.id,
                "description": obj.description,
                "mesh_generated": mesh is not None,
                "vertices_count": len(mesh.vertices) if mesh else 0,
                "faces_count": len(mesh.faces) if mesh else 0,
                "mesh_file": str(mesh.file_path) if mesh and mesh.file_path else None
            }
            summary["objects"].append(obj_summary)
        
        # Save summary as JSON
        import json
        with open(mesh_dir / "hunyuan_generation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved mesh generation summary: {summary['successful_meshes']}/{summary['total_objects']} successful")
