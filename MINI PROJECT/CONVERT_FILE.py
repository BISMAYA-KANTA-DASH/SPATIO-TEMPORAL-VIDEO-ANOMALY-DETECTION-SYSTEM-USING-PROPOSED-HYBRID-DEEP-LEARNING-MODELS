import os
import subprocess

# Folder containing AVI videos
folder_path = "VIDEO"

# Ensure output folder exists
output_folder = os.path.join(folder_path, "converted_mp4")
os.makedirs(output_folder, exist_ok=True)

# Convert each .avi file to .mp4
for file in os.listdir(folder_path):
    if file.endswith(".avi"):
        avi_path = os.path.join(folder_path, file)
        mp4_path = os.path.join(output_folder, file.replace(".avi", ".mp4"))

        # FFmpeg command
        cmd = f'ffmpeg -i "{avi_path}" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k "{mp4_path}"'
        subprocess.run(cmd, shell=True)

print("✅ All AVI files have been converted to MP4!")
