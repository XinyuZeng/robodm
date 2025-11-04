#!/usr/bin/env python3

import os
import random
import subprocess
import re
import sys
from pathlib import Path

def convert_mp4_to_mkv(input_path: str, output_path: str) -> bool:
    cmd = ['ffmpeg', '-i', input_path, '-codec', 'copy', output_path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Conversion failed: {result.stderr.decode()}")
        return False
    return True

if __name__ == "__main__":
    input_path = 'file-000.mp4'
    output_path = 'output.mkv'

    if not convert_mp4_to_mkv(input_path, output_path):
        sys.exit(1)

    abs_mkv = str(Path(output_path).resolve())

    # Get frame count using ffprobe for minimal IO
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0', abs_mkv
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Failed to get frame count")
        sys.exit(1)
    frame_count = int(result.stdout.strip())
    if frame_count == 0:
        print("No frames in video")
        sys.exit(1)

    frame_num = random.randint(0, frame_count - 1)

    code_str = f"""
import av

container = av.open('{abs_mkv}')
video_stream = container.streams.video[0]
timestamp_sec = {frame_num} / video_stream.average_rate
container.seek(int(timestamp_sec * av.time_base), stream=video_stream)
for frame in container.decode(video=0):
    img = frame.to_ndarray(format='bgr24')
    break
container.close()
"""

    log_path = 'io_log.txt'
    cmd = ['strace', '-e', 'trace=open,openat,read', '-o', log_path, 'python3', '-c', code_str]
    subprocess.run(cmd)

    # Parse log
    with open(log_path) as f:
        lines = f.read().splitlines()

    fd = None
    io_sizes = []
    for line in lines:
        if (('openat(' in line or 'open(' in line) and abs_mkv in line
            and not '= -1' in line):  # successful open
            match = re.search(r'=\s*(\d+)', line)
            if match:
                fd = int(match.group(1))
        elif fd is not None and f'read({fd},' in line:
            match = re.search(r'=\s*(\d+)', line)
            if match:
                ret = int(match.group(1))
                if ret > 0:
                    io_sizes.append(ret)

    print(f"Number of IO operations: {len(io_sizes)}")
    print(f"IO sizes: {io_sizes}")
    print(f"Total bytes read: {sum(io_sizes)}")

    os.remove(log_path)
    os.remove(output_path)  # Optional cleanup
