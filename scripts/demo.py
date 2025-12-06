import argparse
import os
import sys

# Fix path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.video_processor import PhobiaVideoProcessor

def main():
    parser = argparse.ArgumentParser(description="PhobiaShield CLI Video Processor")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/videos", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-debug", action="store_true", help="Disable bounding box visualization")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Error: Input file '{args.video}' not found.")
        return

    print(f"🛡️  PhobiaShield CLI Engine Starting...")
    print(f"   Input: {args.video}")
    print(f"   Confidence: {args.conf}")
    
    processor = PhobiaVideoProcessor(output_dir=args.output)
    
    try:
        processor.process_video(
            input_path=args.video,
            output_name=f"processed_{os.path.basename(args.video).split('.')[0]}.webm",
            conf_threshold=args.conf,
            debug=not args.no_debug
        )
    except Exception as e:
        print(f"❌ Critical Failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()