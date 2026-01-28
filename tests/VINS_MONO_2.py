import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob # For listing files easily

# --- Helper function to create a mock image for demonstration (kept for context, but not used for dataset) ---
# ... (create_mock_image function remains the same, but is not called in the main execution block)

# --- Core Functionality: KLT Optical Flow Tracking (Refactored) ---

def process_and_save_tracking(img_a_color, img_b_color, output_path, frame_index):
    """Performs KLT tracking between two images and saves the result."""
    
    if img_a_color is None or img_b_color is None:
        print(f"Skipping frame {frame_index}: One or both images could not be loaded.")
        return False

    # KLT Optical Flow works best on grayscale images
    img_a_gray = cv.cvtColor(img_a_color, cv.COLOR_BGR2GRAY)
    img_b_gray = cv.cvtColor(img_b_color, cv.COLOR_BGR2GRAY)

    # 1. Feature Detection (Frame A) - Shi-Tomasi corners as a stand-in for VINS-Mono features
    feature_params = dict( 
        maxCorners = 100, 
        qualityLevel = 0.3,
        minDistance = 7,
        blockSize = 7
    )
    p0 = cv.goodFeaturesToTrack(img_a_gray, mask = None, **feature_params)
    
    if p0 is None or len(p0) == 0:
        print(f"Warning: No features detected in Frame {frame_index}. Cannot track.")
        return False
        
    # 2. KLT Optical Flow Tracking (Frame A -> Frame B)
    lk_params = dict( 
        winSize  = (15, 15),
        maxLevel = 2,
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # p1 contains the new tracked positions, st is the status array (1=tracked, 0=lost)
    p1, st, err = cv.calcOpticalFlowPyrLK(img_a_gray, img_b_gray, p0, None, **lk_params)

    # 3. Filter out badly tracked points (status == 1)
    # We must flatten the arrays back to usable formats first
    p0_tracked = p0[st == 1]
    p1_tracked = p1[st == 1]
    
    num_tracked = len(p1_tracked)
    # print(f"Frame {frame_index}: Successfully tracked {num_tracked} features.") # Optional printout

    # 4. Visualize and Save the Tracking Results
    
    # Combine the two images side-by-side for drawing
    h1, w1, _ = img_a_color.shape
    h2, w2, _ = img_b_color.shape
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img_a_color
    combined_img[:h2, w1:w1+w2] = img_b_color
    
    # Draw lines connecting the matched points
    for i in range(num_tracked):
        # Draw on Frame A
        cv.circle(combined_img, (int(p0_tracked[i][0]), int(p0_tracked[i][1])), 3, (0, 255, 0), -1) # Green circle on Frame A
        
        # Draw on Frame B (remember to offset the X coordinate)
        x_frame2 = int(p1_tracked[i][0]) + w1
        y_frame2 = int(p1_tracked[i][1])
        cv.circle(combined_img, (x_frame2, y_frame2), 3, (0, 0, 255), -1) # Blue circle on Frame B
        
        # Draw line connecting them (Red line)
        cv.line(combined_img, 
                (int(p0_tracked[i][0]), int(p0_tracked[i][1])), 
                (x_frame2, y_frame2), 
                (255, 0, 0), 1)

    # Save the combined image
    save_filepath = os.path.join(output_path, f'tracking_result_{frame_index:05d}.png')
    cv.imwrite(save_filepath, combined_img)
    
    return True

def vins_visual_tracking_demo():
    # --- Configuration ---
    # NOTE: It's best practice to use raw strings (r'...') for Windows paths to avoid issues with backslashes
    DATASET_PATH = r'C:/Users/Asus/Downloads/VS Studio/New folder/VINS MONO/dataset/image_0'
    OUTPUT_FOLDER = 'C:/Users/Asus/Downloads/VS Studio/New folder/VINS MONO/klt_tracking_results'
    MAX_FRAMES_TO_PROCESS = 200 # Process 200 images, resulting in 199 tracking pairs

    # 1. Setup Output Directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")
    
    # 2. Get a sorted list of image files
    # Assuming images are numbered sequentially (e.g., 000000.png, 000001.png, ...)
    # The glob function finds all files matching the pattern and sort() ensures they are in correct order
    image_files = sorted(glob.glob(os.path.join(DATASET_PATH, '*.png')))
    
    if not image_files:
        print(f"ERROR: No images found in the dataset path: {DATASET_PATH}. Please check the path.")
        return
        
    print(f"Found {len(image_files)} images. Will process up to {MAX_FRAMES_TO_PROCESS} images.")
    
    # Process only up to the limit
    images_to_process = image_files[:MAX_FRAMES_TO_PROCESS]
    
    # 3. Iterate over consecutive image pairs
    # We iterate from the first image up to the second-to-last image
    for i in range(len(images_to_process) - 1):
        img1_path = images_to_process[i]     # Frame A (Current)
        img2_path = images_to_process[i+1]   # Frame B (Next)
        
        # Load the images
        img_a_color = cv.imread(img1_path)
        img_b_color = cv.imread(img2_path)
        
        # Process and save the result for this pair
        # The result is saved using the index 'i', corresponding to Frame A
        success = process_and_save_tracking(img_a_color, img_b_color, OUTPUT_FOLDER, i)
        
        if (i + 1) % 20 == 0 and success:
             print(f"Successfully processed and saved tracking for pair {i} to {i+1}.")
    
    print("\n--- Processing Complete ---")
    print(f"Tracking results for {len(images_to_process) - 1} pairs saved in the '{OUTPUT_FOLDER}' folder.")


if __name__ == '__main__':
    # Ensure the original mock image setup is removed or commented out for the dataset test
    # (The original mock image setup has been removed from the final run block)
    
    # Define a simplified mock creation to avoid errors if the script runs outside the full IDE context
    def create_mock_image(filepath, text):
        pass # Do nothing, we rely on the dataset
    
    vins_visual_tracking_demo()