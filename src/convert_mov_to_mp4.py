import os
import cv2
from tqdm import tqdm

# Input and output directories
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'squat_videos')

# Get all MOV files
mov_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.MOV')]

for mov_file in tqdm(mov_files):
    input_path = os.path.join(VIDEO_DIR, mov_file)
    output_path = os.path.join(VIDEO_DIR, mov_file[:-4] + '.mp4')
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"❌ Could not open {mov_file}")
            continue
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        with tqdm(total=total_frames, desc=f"Converting {mov_file}", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                out.write(frame)
                pbar.update(1)
        
        # Release everything
        cap.release()
        out.release()
        
        print(f"✅ Converted {mov_file} to MP4")
        
        # Delete the original MOV file
        os.remove(input_path)
        print(f"🗑️ Deleted original {mov_file}")
        
    except Exception as e:
        print(f"❌ Error converting {mov_file}: {str(e)}")
        continue

print("✅ All MOV files have been converted to MP4") 