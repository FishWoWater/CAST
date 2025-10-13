"""
Command Line Interface for CAST Pipeline
"""
import argparse
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure to set environment variables manually.")

from cast.core.pipeline import create_pipeline

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CAST: Component-Aligned 3D Scene Reconstruction from RGB Image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cast.cli --image scene.jpg --output ./results
  python -m cast.cli --batch images/ --output ./batch_results
  python -m cast.cli --image scene.jpg --no-intermediates
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        type=str,
        help="Path to input RGB image"
    )
    input_group.add_argument(
        "--batch", "-b",
        type=str,
        help="Directory containing multiple images for batch processing"
    )
    
    # Output arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    # Processing options
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Don't save intermediate results (faster, less disk usage)"
    )
    
    parser.add_argument(
        "--run-id",
        type=str,
        help="Custom run identifier (auto-generated if not provided)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from previous incomplete runs (start fresh)"
    )

    parser.add_argument(
        "--num-max-objects",
        type=int,
        default=-1,
        help="Maximum number of objects to detect (default: 3)"
    )
    
    # Validation options
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate setup without running pipeline"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show interactive visualizations (requires display)"
    )
    
    # Generation options
    parser.add_argument(
        "--enable-generation",
        action="store_true",
        help="Enable Kontext generation for occluded objects"
    )
    
    parser.add_argument(
        "--generation-threshold",
        type=str,
        choices=["no_occlusion",  "some_occlusion", "severe_occlusion"],
        default="some_occlusion",
        help="Minimum occlusion level to trigger generation (default: some_occlusion)"
    )
    
    parser.add_argument(
        "--discard-threshold", 
        type=str,
        choices=["no_occlusion", "some_occlusion", "severe_occlusion"],
        default="severe_occlusion",
        help="Minimum occlusion level to discard objects (default: severe_occlusion)"
    )
    
    parser.add_argument(
        "--generation-provider",
        type=str,
        choices=["replicate", "qwen"],
        default="qwen",
        help="Image generation provider (default: qwen)"
    )
    
    parser.add_argument(
        "--mesh-provider",
        type=str,
        choices=["tripo3d", "trellis", "hunyuan3d"],
        default="trellis",
        help="Mesh provider (default: trellis)"
    )

    parser.add_argument(
        "--mesh-base-url",
        type=str,
        # default="http://192.168.31.42:8080",
        # default="http://127.0.0.1:8080",
        default="",  
        help="Base URL for TRELLIS mesh generation (default: None)"
    )

    parser.add_argument(
        "--pose-estimation-backend",
        type=str,
        choices=["icp", "pytorch"],
        default="icp",
        help="Pose estimation backend (default: icp)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    try:
        pipeline = create_pipeline(output_dir=args.output, 
                                  mesh_provider=args.mesh_provider,
                                  generation_provider=args.generation_provider,
                                  mesh_base_url=args.mesh_base_url,
                                  pose_estimation_backend=args.pose_estimation_backend,
                                  debug=args.debug)
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return 1
    
    if args.validate_only:
        print("Setup validation successful!")
        return 0
    
    # Run pipeline
    try:
        if args.image:
            # Single image processing
            print(f"Processing single image: {args.image}")
            
            result = pipeline.run_single_image(
                image_path=args.image,
                run_id=args.run_id,
                save_intermediates=not args.no_intermediates,
                resume=not args.no_resume,
                num_max_objects=args.num_max_objects,
                enable_generation=args.enable_generation,
                generation_threshold=args.generation_threshold,
                discard_threshold=args.discard_threshold
            )
            
            print(f"Processing complete! Results saved to: {result.output_dir}")
            
        elif args.batch:
            # Batch processing
            batch_dir = Path(args.batch)
            if not batch_dir.exists():
                print(f"Error: Batch directory {batch_dir} does not exist")
                return 1
            
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = [
                str(f) for f in batch_dir.rglob("*") 
                if f.suffix.lower() in image_extensions
            ]
            
            if not image_files:
                print(f"Error: No image files found in {batch_dir}")
                return 1
            
            print(f"Found {len(image_files)} images for batch processing")
            
            results = pipeline.run_batch(
                image_paths=image_files,
                base_output_dir=Path(args.output),
                enable_generation=args.enable_generation,
                generation_threshold=args.generation_threshold,
                discard_threshold=args.discard_threshold
            )
            
            print(f"Batch processing complete! {len(results)} successful results.")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())