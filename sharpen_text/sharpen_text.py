import cv2
import numpy as np

def unsharp_mask_text(image_path, blur_sigma=1.0, sharpen_amount=3.0):
    """
    Applies the Unsharp Mask sharpening technique.
    
    The Unsharp Mask works by subtracting a blurred version of the image
    from the original to isolate edges, then adding the scaled edges back.
    
    :param image_path: Path to the input image file.
    :param blur_sigma: Standard deviation for Gaussian blur. Controls the size of details to enhance.
                       A lower value (e.g., 1.0) is better for fine details like text.
    :param sharpen_amount: The intensity/strength of the sharpening effect (the 'boost').
    :return: The sharpened image as a NumPy array, or None if the image fails to load.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None

    # 2. Create the unsharp (blurred) image using Gaussian Blur
    # The kernel size (0, 0) tells OpenCV to calculate it automatically based on sigma.
    blurred = cv2.GaussianBlur(img, (0, 0), blur_sigma)

    # 3. Apply the Unsharp Mask formula using cv2.addWeighted:
    # sharpened = original * (1 + amount) + blurred * (-amount)
    sharpened = cv2.addWeighted(
        img, 
        1.0 + sharpen_amount,  # Alpha weight for the original image
        blurred, 
        -sharpen_amount,      # Beta weight for the blurred image (subtractive mask)
        0                     # Gamma offset
    )

    return sharpened


# --- Main Execution ---
if __name__ == "__main__":
    
    # 📢 --- 1. SET THE FILE PATH ---
    # Example path based on your previous input:
    input_file = "IMAGE_FILE/bimg.jpg" 
    
    # --- 2. TUNE THESE PARAMETERS FOR TEXT ---
    # Adjust these values for the best result on your blurry text.
    SIGMA = 1.0     # Best range for text: 0.5 to 2.0. Keeps sharpening focused on fine details.
    AMOUNT = 3.0    # Best range for text: 2.5 to 5.0. Controls how aggressively contrast is boosted.
    
    output_file = "sharpened_output.jpg" 

    sharpened_image = unsharp_mask_text(input_file, SIGMA, AMOUNT)

    if sharpened_image is not None:
        # Save the resulting image
        cv2.imwrite(output_file, sharpened_image)
        print(f"✅ Image successfully sharpened and saved as: {output_file}")
        
        # Display the result (requires a windowing environment)
        cv2.imshow("Original Image", cv2.imread(input_file))
        cv2.imshow("Sharpened Image", sharpened_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Sharpening failed due to image loading error.")