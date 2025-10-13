"""
Main CAST Pipeline

This module orchestrates the complete Component-Aligned 3D Scene Reconstruction pipeline.
"""
import numpy as np
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path
import time
import json
import pickle
from datetime import datetime

from ..modules.detection_segmentation import DetectionSegmentationModule
from ..modules.detection_filtering import DetectionFilteringModule
from ..modules.image_generation import ImageGenerationModule
from ..modules.depth_estimation import DepthEstimationModule
from ..modules.mesh_generation import MeshGenerationModule
from ..modules.hunyuan_mesh_generation import HunyuanMeshGenerationModule
from ..modules.pose_estimation import PoseEstimationModule
from ..modules.scene_graph_optimization import SceneGraphOptimizationModule

from .common import SceneReconstruction, OCCLUSION_LEVELS
from ..utils.image_utils import load_image, save_image
from ..config.settings import config

class CASTPipeline:
    """
    Complete CAST (Component-Aligned 3D Scene Reconstruction) Pipeline
    
    This pipeline takes a single RGB image and produces:
    - Detected and segmented objects
    - Depth map and point cloud
    - 3D meshes for each object
    - 6D poses for each object
    - Scene graph with object relationships
    - Physically plausible scene reconstruction
    """
    
    def __init__(self, output_dir: Optional[Path] = None, 
                 mesh_provider: Literal["tripo3d", "trellis", "hunyuan3d"] = "trellis",
                 mesh_base_url: Optional[str] = None,
                 generation_provider: Literal["replicate", "qwen"] = "qwen", 
                 pose_estimation_backend: Literal["icp", "pytorch"] = "icp",
                 enable_render_and_compare: bool = False,
                 enable_scene_graph_opt: bool = False, 
                 debug: bool = False):
        """
        Initialize the CAST pipeline
        
        Args:
            output_dir: Directory to save all outputs
            mesh_provider: 3D mesh generation provider ("tripo3d", "trellis", or "hunyuan3d")
            generation_provider: Image generation provider ("replicate" or "qwen")
            pose_estimation_backend: Backend for pose estimation ("icp" or "pytorch")
            enable_render_and_compare: Whether to enable render-and-compare for pose estimation
            enable_scene_graph_opt: Whether to enable scene graph optimization
            debug: Whether to enable debug mode
        """
        self.output_dir = output_dir or config.paths.output_dir
        self.mesh_provider = mesh_provider
        
        # Initialize all modules
        print("Initializing CAST pipeline modules...")
        self.detection_module = DetectionSegmentationModule()
        self.filtering_module = DetectionFilteringModule()
        self.generation_module = ImageGenerationModule(provider=generation_provider)
        self.depth_module = DepthEstimationModule()
        
        # Initialize mesh generation module based on provider
        if mesh_provider == "hunyuan3d":
            self.mesh_module = HunyuanMeshGenerationModule()
        else:
            self.mesh_module = MeshGenerationModule(provider=mesh_provider, base_url=mesh_base_url)
        
        self.pose_module = PoseEstimationModule(backend=pose_estimation_backend,  enable_render_and_compare=enable_render_and_compare, debug=debug)
        self.scene_graph_module = SceneGraphOptimizationModule() if enable_scene_graph_opt else None
        self.debug = debug
        
        print("CAST pipeline initialized successfully")
    
    def validate_setup(self) -> bool:
        """Validate that all required API keys and dependencies are available"""
        print("Validating pipeline setup...")
        
        # Check API configuration
        if not config.validate():
            print("Error: Missing required API keys. Please check your .env file.")
            return False
        
        # Check if MoGe model can be loaded
        try:
            if not self.depth_module.load_model():
                print("Error: Failed to load MoGe model")
                return False
        except Exception as e:
            print(f"Error validating MoGe setup: {e}")
            return False
        
        print("Pipeline setup validation complete")
        return True
    
    def _save_stage_result(self, stage_name: str, result: Any, output_dir: Path) -> None:
        """Save intermediate stage result for resuming"""
        stage_file = output_dir / f"{stage_name}_result.pkl"
        try:
            with open(stage_file, 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved {stage_name} stage result")
        except Exception as e:
            print(f"Warning: Failed to save {stage_name} stage result: {e}")
    
    def _load_stage_result(self, stage_name: str, output_dir: Path) -> Optional[Any]:
        """Load intermediate stage result for resuming"""
        stage_file = output_dir / f"{stage_name}_result.pkl"
        if stage_file.exists():
            try:
                with open(stage_file, 'rb') as f:
                    result = pickle.load(f)
                print(f"Loaded {stage_name} stage result from previous run")
                return result
            except Exception as e:
                print(f"Warning: Failed to load {stage_name} stage result: {e}")
        return None
    
    def _stage_completed(self, stage_name: str, output_dir: Path) -> bool:
        """Check if a stage has been completed successfully"""
        stage_file = output_dir / f"{stage_name}_result.pkl"
        return stage_file.exists()
    
    def _create_stage_marker(self, stage_name: str, output_dir: Path) -> None:
        """Create a marker file to indicate stage completion"""
        marker_file = output_dir / f"{stage_name}_completed.marker"
        marker_file.touch()
        
    def _get_completed_stages(self, output_dir: Path) -> List[str]:
        """Get list of completed stages"""
        completed = []
        stages = ['detection', 'filtering', 'depth', 'generation', 'mesh', 'pose', 'scene_graph']
        for stage in stages:
            if self._stage_completed(stage, output_dir):
                completed.append(stage)
        return completed
    

    
    def run_single_image(self, image_path: str, 
                        run_id: Optional[str] = None,
                        save_intermediates: bool = True,
                        resume: bool = True,
                        num_max_objects: int = -1,
                        enable_qwen_filtering: bool = True,
                        enable_size_filtering: bool = True,
                        enable_occlusion_filtering: bool = True,
                        min_area_ratio: float = 0.001,
                        max_area_ratio: float = 0.8,
                        enable_fitness_filtering: bool = True,
                        min_fitness_threshold: float = 0.1,
                        enable_generation: bool = False,
                        generation_threshold: str = "some_occlusion",
                        discard_threshold: str = "severe_occlusion",
                        **mesh_kwargs) -> SceneReconstruction:
        """
        Run the complete CAST pipeline on a single image
        
        Args:
            image_path: Path to input image
            run_id: Optional run identifier for organizing outputs
            save_intermediates: Whether to save intermediate results
            resume: Whether to resume from previous incomplete runs
            num_max_objects: Maximum number of objects to detect
            enable_qwen_filtering: Whether to use Qwen-VL filtering
            enable_size_filtering: Whether to use size-based filtering
            enable_occlusion_filtering: Whether to use occlusion-based filtering
            min_area_ratio: Minimum area ratio for size filtering
            max_area_ratio: Maximum area ratio for size filtering
            enable_fitness_filtering: Whether to filter by pose fitness
            min_fitness_threshold: Minimum fitness threshold for pose filtering
            enable_generation: Whether to enable Kontext generation for occluded objects
            generation_threshold: Minimum occlusion level to trigger generation
            discard_threshold: Minimum occlusion level to discard objects
            **mesh_kwargs: Additional parameters for mesh generation (e.g., TRELLIS parameters)
            
        Returns:
            Complete scene reconstruction result
        """
        start_time = time.time()
        
        # Use image basename for output directory instead of timestamp
        image_name = Path(image_path).stem
        if run_id is None:
            run_output_dir = self.output_dir / image_name
        else:
            run_output_dir = self.output_dir / f"{image_name}_{run_id}"
        
        run_output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Starting CAST pipeline for image: {image_path}")
        print(f"Output directory: {run_output_dir}")
        
        # Check for resume capability
        completed_stages = []
        if resume:
            completed_stages = self._get_completed_stages(run_output_dir)
            if completed_stages:
                print(f"Found previous run with completed stages: {', '.join(completed_stages)}")
        
        # Load input image
        try:
            image = load_image(image_path)
            print(f"Loaded image: {image.shape}")
            
            # Save input image to output directory
            save_image(image, run_output_dir / "input_image.png")
            
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
        
        # Step 1: Object Detection and Segmentation
        print("\n" + "="*60)
        print("STEP 1: Object Detection and Segmentation")
        print("="*60)
        
        if resume and 'detection' in completed_stages:
            print("Resuming: Loading detection results from previous run...")
            detected_objects = self._load_stage_result('detection', run_output_dir)
        else:
            detected_objects = self.detection_module.run(
                image, 
                output_dir=run_output_dir if save_intermediates else None
            )
            # only when there are detected objects we save the results 
            if detected_objects and save_intermediates:
                self._save_stage_result('detection', detected_objects, run_output_dir)
        
        if not detected_objects:
            raise RuntimeError("No objects detected in the image")
        
        # Step 1.5: Detection Filtering (Optional)
        print("\n" + "="*60)
        print("STEP 1.5: Detection Filtering")
        print("="*60)
        
        if resume and 'filtering' in completed_stages:
            print("Resuming: Loading filtering results from previous run...")
            detected_objects = self._load_stage_result('filtering', run_output_dir)
        else:
            # Apply detection filtering to remove spurious detections
            detected_objects = self.filtering_module.run(
                image, 
                detected_objects,
                use_qwen_filter=enable_qwen_filtering,
                use_size_filter=enable_size_filtering,
                use_occlusion_filter=enable_occlusion_filtering,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                occlusion_threshold=discard_threshold,
                output_dir=run_output_dir if save_intermediates else None
            )
            if detected_objects and save_intermediates:
                self._save_stage_result('filtering', detected_objects, run_output_dir)
        
        if not detected_objects:
            raise RuntimeError("No objects remaining after filtering")

        # if num_max_objects > 0:
        #     detected_objects = detected_objects[:num_max_objects]
        #     print(f"Filtered to {len(detected_objects)} objects due to maximum number of objects")
        
        # Step 2: Depth Estimation
        print("\n" + "="*60)
        print("STEP 2: Depth Estimation")
        print("="*60)
        
        if resume and 'depth' in completed_stages:
            print("Resuming: Loading depth estimation results from previous run...")
            depth_estimation = self._load_stage_result('depth', run_output_dir)
        else:
            depth_estimation = self.depth_module.run(
                image,
                output_dir=run_output_dir if save_intermediates else None,
                detected_objects=detected_objects  # Pass detected objects for instance point clouds
            )
            if depth_estimation is not None and save_intermediates:
                self._save_stage_result('depth', depth_estimation, run_output_dir)
        
        # Step 2.5: Object Generation (Optional)
        if enable_generation:
            print("\n" + "="*60)
            print("STEP 2.5: Object Generation")
            print("="*60)
            
            if resume and 'generation' in completed_stages:
                print("Resuming: Loading generation results from previous run...")
                detected_objects = self._load_stage_result('generation', run_output_dir)
            else:
                # Generate images for occluded objects
                detected_objects = self.generation_module.run(
                    detected_objects, OCCLUSION_LEVELS.get(generation_threshold, 1), 
                    output_dir=run_output_dir if save_intermediates else None
                )
                
                if save_intermediates:
                    self._save_stage_result('generation', detected_objects, run_output_dir)
                    self._save_generation_results(detected_objects, run_output_dir)

        
        # Step 3: 3D Mesh Generation
        print("\n" + "="*60)
        print("STEP 3: 3D Mesh Generation")
        print("="*60)

        # exit()
        if num_max_objects > 0:
            detected_objects = detected_objects[:num_max_objects]
            print(f"Filtered to {len(detected_objects)} objects due to maximum number of objects")
        
        if resume and 'mesh' in completed_stages:
            print("Resuming: Loading mesh generation results from previous run...")
            mesh_data = self._load_stage_result('mesh', run_output_dir)
            meshes, valid_objects = mesh_data['meshes'], mesh_data['objects']
        else:
            # For Hunyuan3D, pass depth_estimation for point cloud conditioning
            if self.mesh_provider == "hunyuan3d":
                meshes = self.mesh_module.run(
                    detected_objects,
                    depth_estimation,
                    output_dir=run_output_dir if save_intermediates else None
                )
            else:
                meshes = self.mesh_module.run(
                    detected_objects,
                    output_dir=run_output_dir if save_intermediates else None,
                    **mesh_kwargs
                )
            
            # Filter out failed mesh generations
            valid_pairs = [(mesh, obj) for mesh, obj in zip(meshes, detected_objects) if mesh is not None]
            
            if not valid_pairs:
                raise RuntimeError("No successful mesh generations")
            
            meshes, valid_objects = zip(*valid_pairs)
            meshes, valid_objects = list(meshes), list(valid_objects)
            
            if len(meshes) >  0 and len(valid_objects) > 0 and save_intermediates:
                mesh_data = {'meshes': meshes, 'objects': valid_objects}
                self._save_stage_result('mesh', mesh_data, run_output_dir)
        
        # Step 4: Pose Estimation
        print("\n" + "="*60)
        print("STEP 4: Pose Estimation")
        print("="*60)
        
        if resume and 'pose' in completed_stages:
            print("Resuming: Loading pose estimation results from previous run...")
            objects_3d = self._load_stage_result('pose', run_output_dir)
        else:
            objects_3d = self.pose_module.run(
                meshes,
                valid_objects,
                depth_estimation,
                output_dir=run_output_dir if save_intermediates else None,
                filter_low_fitness=enable_fitness_filtering,  # Enable fitness-based filtering
                min_fitness_threshold=min_fitness_threshold,  # Minimum fitness threshold
                image=image  # Pass image for instance point cloud extraction
            )
            if objects_3d is not None and save_intermediates:
                self._save_stage_result('pose', objects_3d, run_output_dir)
        
        # Step 5: Scene Graph Extraction and Optimization
        print("\n" + "="*60)
        print("STEP 5: Scene Graph Extraction and Optimization")
        print("="*60)
        
        if self.scene_graph_module is not None:
            if resume and 'scene_graph' in completed_stages:
                print("Resuming: Loading scene graph results from previous run...")
                scene_data = self._load_stage_result('scene_graph', run_output_dir)
                scene_graph, objects_3d = scene_data['scene_graph'], scene_data['objects']
            else:
                scene_graph, objects_3d = self.scene_graph_module.run(
                    image,
                    objects_3d,
                    output_dir=run_output_dir if save_intermediates else None
                )
                if save_intermediates:
                    scene_data = {'scene_graph': scene_graph, 'objects': objects_3d}
                    self._save_stage_result('scene_graph', scene_data, run_output_dir)
        else:
            scene_graph = None
            
        # Create final reconstruction result
        reconstruction = SceneReconstruction(
            input_image=image,
            detected_objects=detected_objects,
            depth_estimation=depth_estimation,
            objects_3d=objects_3d,
            scene_graph=scene_graph,
            output_dir=run_output_dir
        )
        
        # Save final results
        self._save_final_results(reconstruction, run_output_dir)
        
        # Print summary
        total_time = time.time() - start_time
        self._print_summary(reconstruction, total_time)
        
        return reconstruction
    
    def _save_final_results(self, reconstruction: SceneReconstruction, 
                           output_dir: Path) -> None:
        """Save final reconstruction results"""
        print("\nSaving final reconstruction results...")
        
        # Save reconstruction summary
        summary = reconstruction.save_summary()
        with open(output_dir / "reconstruction_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save final scene visualization
        try:
            self._create_scene_visualization(reconstruction, output_dir)
        except Exception as e:
            print(f"Warning: Failed to create scene visualization: {e}")
        
        print(f"Final results saved to: {output_dir}")
    
    def _create_scene_visualization(self, reconstruction: SceneReconstruction, 
                                  output_dir: Path) -> None:
        """Create visualization of the complete reconstructed scene using trimesh.Scene"""
        try:
            import trimesh
            import numpy as np
            
            # Create trimesh scene
            scene = trimesh.Scene()
            
            # Add scene point cloud if available
            # if reconstruction.depth_estimation.point_cloud is not None:
            #     # Convert point cloud to trimesh format
            #     points = reconstruction.depth_estimation.point_cloud
            #     if len(points.shape) == 3:  # If it's an image-shaped array
            #         points = points.reshape(-1, 3)
                
            #     # Remove invalid points
            #     valid_mask = ~np.any(np.isnan(points) | (points == 0), axis=1)
            #     valid_points = points[valid_mask]
                
            #     if len(valid_points) > 0:
            #         # Create point cloud mesh
            #         point_cloud = trimesh.points.PointCloud(valid_points)
            #         scene.add_geometry(point_cloud, node_name="scene_point_cloud")
            
            # Add each reconstructed object with proper pose transformation
            for i, obj in enumerate(reconstruction.objects_3d):
                try:
                    # Create trimesh object
                    if obj.mesh.file_path and obj.mesh.file_path.exists():
                        # Load from GLB file if available (preserves materials/textures)
                        mesh = trimesh.load(obj.mesh.file_path)
                        if isinstance(mesh, trimesh.Scene):
                            # If it's a scene, get the first mesh
                            mesh = mesh.dump(concatenate=True)
                    else:
                        # Create from vertices and faces
                        mesh = trimesh.Trimesh(
                            vertices=obj.mesh.vertices,
                            faces=obj.mesh.faces
                        )
                    
                    # Create transformation matrix with scale support
                    transform = np.eye(4)
                    
                    # Handle scale if available
                    if hasattr(obj.pose, 'scale') and obj.pose.scale is not None:
                        # Apply scale to rotation matrix
                        scaled_rotation = obj.pose.rotation @ np.diag(obj.pose.scale)
                        transform[:3, :3] = scaled_rotation
                    else:
                        transform[:3, :3] = obj.pose.rotation
                    
                    transform[:3, 3] = obj.pose.translation
                    
                    # Apply transformation to mesh
                    mesh.apply_transform(transform)
                    
                    # Add to scene with descriptive name
                    node_name = f"object_{obj.id}_{obj.detected_object.description.replace(' ', '_')}"
                    scene.add_geometry(mesh, geom_name=node_name)
                    
                    print(f"Added object {obj.id} ({obj.detected_object.description}) to scene")
                    
                except Exception as e:
                    print(f"Warning: Failed to add object {obj.id} to scene: {e}")
                    continue
            
            # Save the complete scene
            scene_file = output_dir / "reconstructed_scene.glb"
            scene.export(str(scene_file))
            print(f"Complete scene saved to: {scene_file}")
            
            # Also save as PLY for point cloud compatibility
            try:
                scene_ply = output_dir / "reconstructed_scene.ply"
                combined_mesh = scene.dump(concatenate=True)
                combined_mesh.export(str(scene_ply))
                print(f"Scene also saved as PLY: {scene_ply}")
            except Exception as e:
                print(f"Warning: Could not save PLY version: {e}")
            
            # Save individual object meshes with poses applied
            objects_dir = output_dir / "positioned_objects"
            objects_dir.mkdir(exist_ok=True)
            
            for obj in reconstruction.objects_3d:
                try:
                    # Create individual mesh file
                    if obj.mesh.file_path and obj.mesh.file_path.exists():
                        mesh = trimesh.load(obj.mesh.file_path)
                        if isinstance(mesh, trimesh.Scene):
                            mesh = mesh.dump(concatenate=True)
                    else:
                        mesh = trimesh.Trimesh(vertices=obj.mesh.vertices, faces=obj.mesh.faces)
                    
                    # Apply pose transformation
                    transform = np.eye(4)
                    if hasattr(obj.pose, 'scale') and obj.pose.scale is not None:
                        scaled_rotation = obj.pose.rotation @ np.diag(obj.pose.scale)
                        transform[:3, :3] = scaled_rotation
                    else:
                        transform[:3, :3] = obj.pose.rotation
                    transform[:3, 3] = obj.pose.translation
                    
                    mesh.apply_transform(transform)
                    
                    # Save individual object
                    obj_file = objects_dir / f"object_{obj.id}_positioned.glb"
                    mesh.export(str(obj_file))
                    
                except Exception as e:
                    print(f"Warning: Could not save individual object {obj.id}: {e}")
            
            print(f"Individual positioned objects saved to: {objects_dir}")
            
        except Exception as e:
            print(f"Error creating scene visualization: {e}")
    
    def _print_summary(self, reconstruction: SceneReconstruction, total_time: float) -> None:
        """Print pipeline execution summary"""
        print("\n" + "="*80)
        print("CAST PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Objects detected: {len(reconstruction.detected_objects)}")
        print(f"3D objects reconstructed: {len(reconstruction.objects_3d)}")
        if reconstruction.scene_graph is not None:
            print(f"Scene relationships: {len(reconstruction.scene_graph.relationships)}")
        
        print("\nDetected Objects:")
        for obj in reconstruction.detected_objects:
            print(f"  - Object {obj.id}: {obj.description} (confidence: {obj.confidence:.3f})")
        
        if reconstruction.scene_graph is not None:
            print("\nScene Relationships:")
            for rel in reconstruction.scene_graph.relationships:
                pair = rel.get('pair', [])
                rel_type = rel.get('relationship', 'Unknown')
                reason = rel.get('reason', 'No reason provided')
                print(f"  - Objects {pair}: {rel_type} ({reason})")
        
        print(f"\nResults saved to: {reconstruction.output_dir}")
        print("="*80)
    
    def _save_generation_results(self, detected_objects: List, output_dir: Path) -> None:
        """Save generation results for analysis"""
        generation_dir = output_dir / "generation"
        generation_dir.mkdir(exist_ok=True, parents=True)
        
        # Save generation summary
        generation_summary = {
            "total_objects": len(detected_objects),
            "generated_objects": [
                {
                    "id": obj.id,
                    "description": obj.description,
                    "occlusion_level": getattr(obj, 'occlusion_level', 'no_occlusion'),
                    "generated": obj.generated_image is not None
                }
                for obj in detected_objects
            ]
        }
        
        with open(generation_dir / "generation_summary.json", "w") as f:
            json.dump(generation_summary, f, indent=2)
        
        print(f"Generation results saved to: {generation_dir}")

    def run_batch(self, image_paths: list, 
                 base_output_dir: Optional[Path] = None,
                 enable_generation: bool = False,
                 generation_threshold: str = "some_occlusion",
                 discard_threshold: str = "severe_occlusion",
                 **pipeline_kwargs) -> Dict[str, SceneReconstruction]:
        """
        Run CAST pipeline on multiple images
        
        Args:
            image_paths: List of image file paths
            base_output_dir: Base directory for outputs
            enable_generation: Whether to enable Kontext generation for occluded objects
            generation_threshold: Minimum occlusion level to trigger generation
            discard_threshold: Minimum occlusion level to discard objects
            enable_render_and_compare: Whether to enable render-and-compare for pose estimation
            **pipeline_kwargs: Additional pipeline parameters
            
        Returns:
            Dictionary mapping image paths to reconstruction results
        """
        results = {}
        
        if base_output_dir:
            self.output_dir = base_output_dir
        
        print(f"Running CAST pipeline on {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # Generate unique run ID
                run_id = f"batch_{i:03d}_{Path(image_path).stem}"
                
                # Run pipeline with resume enabled by default
                result = self.run_single_image(
                    image_path, 
                    run_id=run_id, 
                    resume=True,
                    enable_generation=enable_generation,
                    generation_threshold=generation_threshold,
                    discard_threshold=discard_threshold,
                    **pipeline_kwargs
                )
                results[image_path] = result
                
                print(f"Successfully processed {image_path}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"\nBatch processing complete. {len(results)}/{len(image_paths)} successful.")
        return results

def create_pipeline(output_dir: Optional[str] = None, 
                   mesh_provider: Literal["tripo3d", "trellis", "hunyuan3d"] = "trellis",
                   mesh_base_url: Optional[str] = None,
                   generation_provider: Literal["replicate", "qwen"] = "replicate", 
                   pose_estimation_backend: Literal["icp", "pytorch"] = "icp",
                   debug: bool = False) -> CASTPipeline:
    """
    Factory function to create a CAST pipeline
    
    Args:
        output_dir: Output directory path
        mesh_provider: 3D mesh generation provider ("tripo3d", "trellis", or "hunyuan3d")
        generation_provider: Image generation provider ("replicate" or "qwen")
        
    Returns:
        Initialized CAST pipeline
    """
    output_path = Path(output_dir) if output_dir else None
    pipeline = CASTPipeline(output_dir=output_path, 
                           mesh_provider=mesh_provider,
                           mesh_base_url=mesh_base_url,
                           generation_provider=generation_provider,
                           pose_estimation_backend=pose_estimation_backend,
                           debug=debug)
    
    # Validate setup
    if not pipeline.validate_setup():
        raise RuntimeError("Pipeline setup validation failed")
    
    return pipeline