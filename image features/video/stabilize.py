import cv2
import numpy as np

def stabilize_video(input_video, output_video, max_corners=1000, smooth_radius=30):
    """
    Stabilize video using ORB features and optical flow.
    
    Args:
        input_video (str): Path to input video file
        output_video (str): Path to save stabilized video
        max_corners (int): Maximum number of ORB features to track
        smooth_radius (int): Radius for smoothing camera motion (larger = smoother)
    """
    # Initialize video capture
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=max_corners)
    
    # Read first frame
    success, prev_frame = cap.read()
    if not success:
        print("Error reading first frame")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features in first frame
    prev_kps = orb.detect(prev_gray, None)
    prev_pts = cv2.KeyPoint_convert(prev_kps)
    prev_pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)
    
    # Initialize transformations array
    transforms = np.zeros((frame_count-1, 3), np.float32)
    
    # Calculate motion between frames
    for i in range(frame_count-1):
        # Read next frame
        success, curr_frame = cap.read()
        if not success:
            break
        
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Select good points
        idx = np.where(status == 1)[0]
        prev_pts_refined = prev_pts[idx]
        curr_pts_refined = curr_pts[idx]
        
        # Estimate affine transformation
        if len(prev_pts_refined) >= 4:
            transform, _ = cv2.estimateAffinePartial2D(
                prev_pts_refined, curr_pts_refined)
            
            if transform is not None:
                # Extract translation and rotation
                dx = transform[0, 2]
                dy = transform[1, 2]
                da = np.arctan2(transform[1, 0], transform[0, 0])
            else:
                dx, dy, da = 0, 0, 0
        else:
            dx, dy, da = 0, 0, 0
        
        transforms[i] = [dx, dy, da]
        
        # Update previous frame and points
        prev_gray = curr_gray.copy()
        prev_pts = curr_pts_refined.reshape(-1, 1, 2)
        
        # Periodically redetect ORB features
        if len(prev_pts) < max_corners//2:
            prev_kps = orb.detect(prev_gray, None)
            prev_pts = cv2.KeyPoint_convert(prev_kps)
            prev_pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)
    
    # Compute cumulative motion
    trajectory = np.cumsum(transforms, axis=0)
    
    # Smooth the trajectory
    smoothed_trajectory = np.zeros_like(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = np.convolve(
            trajectory[:, i], np.ones(smooth_radius)/smooth_radius, mode='same')
    
    # Calculate difference between original and smoothed trajectory
    difference = smoothed_trajectory - trajectory
    
    # Add back the difference to transforms
    transforms_smooth = transforms + difference
    
    # Reset video capture to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Apply stabilization
    for i in range(frame_count-1):
        # Read frame
        success, frame = cap.read()
        if not success:
            break
        
        # Extract transformation parameters
        dx, dy, da = transforms_smooth[i]
        
        # Build transformation matrix
        transform = np.zeros((2, 3), np.float32)
        transform[0, 0] = np.cos(da)
        transform[0, 1] = -np.sin(da)
        transform[1, 0] = np.sin(da)
        transform[1, 1] = np.cos(da)
        transform[0, 2] = dx
        transform[1, 2] = dy
        
        # Apply affine transformation
        stabilized_frame = cv2.warpAffine(
            frame, transform, (width, height),
            borderMode=cv2.BORDER_REPLICATE)
        
        # Write stabilized frame
        out.write(stabilized_frame)
        
        # Display progress
        if i % 10 == 0:
            print(f"Processing frame {i+1}/{frame_count}")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Stabilized video saved to {output_video}")

if __name__ == "__main__":
    # Example usage
    input_video = "short2.mp4"
    output_video = "stabilized_video.mp4"
    
    stabilize_video(
        input_video=input_video,
        output_video=output_video,
        max_corners=1000,
        smooth_radius=30
    )