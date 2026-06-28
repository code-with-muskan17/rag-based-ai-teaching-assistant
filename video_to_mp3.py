import os
import subprocess

files = os.listdir("videos")

for file in files:
    tutorial_no = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" ｜ ")[0]
    print(f"Converting: {file}")
    subprocess.run([
        "ffmpeg", "-i",
        f"videos/{file}",
        f"audios/{tutorial_no}_{file_name}.mp3"
    ])
    print(f"Saved: audios/{tutorial_no}_{file_name}.mp3")