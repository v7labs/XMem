import cv2
import os

def images_to_video(input_folder, output_file, fps):
    # Get the list of image filenames in the input folder

    breakpoint()
    
    image_files = sorted(os.listdir(input_folder))
    images = []
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        if img is not None:
            images.append(img)

    # Get the dimensions of the first image
    height, width, _ = images[0].shape

    # Define the video codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create the VideoWriter object
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Write each image to the video file
    for img in images:
        video.write(img)

    # Release the VideoWriter object
    video.release()

    print(f"Video created: {output_file}")


# Specify the input folder containing the images
input_folder = '/workspaces/data/tom/Totalsegmentator_MSD_format/Task_vertebrae_T4/480pVolumetricallyExtractedData/JPEGImages/0864'

# Specify the output video file
output_file = 't4.mp4'

# Specify the desired frame rate of the output video
fps = 30

# Call the function to convert images to video
images_to_video(input_folder, output_file, fps)
