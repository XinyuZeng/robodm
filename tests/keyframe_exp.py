import subprocess
import os
import time
import random
import csv
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Keyframe interval experiment for video codecs')
parser.add_argument('--force', action='store_true', help='Force regeneration of files even if they already exist')
parser.add_argument('--output', '-o', type=str, default='data/results.csv', help='Output CSV file path (default: data/results.csv)')
parser.add_argument('--input-file', '-i', type=str, default='data/file-000.mp4', help='Input video file path (default: data/file-000.mp4)')
args = parser.parse_args()

input_file = args.input_file  # Input file path

codecs = {
    'av1': {'lib': 'libaom-av1', 
            # 'presets': [0, 4, 8]# cpu-used values (slow to fast)
            'presets': [8]# cpu-used values (slow to fast)
            },  
    # 'h265': {'lib': 'libx265', 
    #         #  'presets': ['medium', 'fast', 'ultrafast']
    #         'presets': ['fast','ultrafast']
    #         }
}

keyframe_intervals = [5, 10, 20, 50, 100, 200, 300, 400, 500]  # Example intervals; adjust as needed

output_files = []

# Extract base name from input file (without extension)
input_basename = os.path.splitext(os.path.basename(input_file))[0]

# Step 1: Encode all videos
for codec_name, info in codecs.items():
    codec = info['lib']
    for preset in info['presets']:
        for interval in keyframe_intervals:
            output_file = f"data/{input_basename}-keyframe{interval}-{codec_name}-preset{preset}.mkv"
            
            # Check if file already exists
            if os.path.exists(output_file) and not args.force:
                print(f"Skipping {output_file} (already exists, use --force to regenerate)")
                # Still need to measure encode time, so we'll set it to 0 for existing files
                encode_time = 0
            else:
                # Encode the video with specified keyframe interval
                encode_cmd = [
                    'ffmpeg', '-i', input_file, 
                    '-c:v', codec, '-g', str(interval)
                ]
                if codec_name == 'av1':
                    encode_cmd.extend(['-cpu-used', str(preset)])
                elif codec_name == 'h265':
                    encode_cmd.extend(['-preset', str(preset)])
                encode_cmd.extend(['-y', output_file])
                print(f"Encoding {output_file}...")
                start = time.time()
                subprocess.run(encode_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                encode_time = time.time() - start
            
            output_files.append({
                'file': output_file,
                'codec': codec_name,
                'preset': preset,
                'interval': interval,
                'encode_time': encode_time
            })

# Step 2: Measure for each output file
results = []
duration_cmd = [
    'ffprobe', '-v', 'error', 
    '-show_entries', 'format=duration', 
    '-of', 'default=noprint_wrappers=1:nokey=1', 
    input_file
]
duration = float(subprocess.check_output(duration_cmd))

for item in output_files:
    output_file = item['file']
    codec_name = item['codec']
    preset = item['preset']
    interval = item['interval']
    
    # Get file size in MB
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    access_times = []
    for _ in range(10):
        # Clear OS page cache before each random access (requires sudo)
        subprocess.run(['sync'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Choose a random timestamp
        random_time = random.uniform(0, duration)
        
        # Measure time to seek and access one frame
        start = time.time()
        seek_cmd = [
            'ffmpeg', '-ss', str(random_time), '-i', output_file, 
            '-frames:v', '1', '-f', 'null', '/dev/null'
        ]
        subprocess.run(seek_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        access_time = time.time() - start
        access_times.append(access_time)
    
    avg_access_time = sum(access_times) / len(access_times)
    
    # Measure time to read the whole mkv file
    # Clear OS page cache before reading (requires sudo)
    subprocess.run(['sync'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    start = time.time()
    read_whole_cmd = [
        'ffmpeg', '-i', output_file, 
        '-f', 'null', '/dev/null'
    ]
    subprocess.run(read_whole_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    whole_file_read_time = time.time() - start
    
    results.append({
        'codec': codec_name,
        'preset': preset,
        'interval': interval,
        'size_mb': size_mb,
        'access_time': avg_access_time,
        'whole_file_read_time': whole_file_read_time,
        'encode_time': item['encode_time']
    })
    print(f"Measured {output_file}: Size {size_mb:.2f} MB, Avg Access Time {avg_access_time:.4f} s, Whole File Read Time {whole_file_read_time:.4f} s, Encode Time {item['encode_time']:.2f} s")

# Step 3: Output to CSV
csv_file = args.output
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Codec', 'Preset', 'Interval', 'Size_MB', 'Access_Time_s', 'Whole_File_Read_Time_s', 'Encode_Time_s'])
    for res in results:
        writer.writerow([res['codec'], res['preset'], res['interval'], res['size_mb'], res['access_time'], res['whole_file_read_time'], res['encode_time']])

print(f"Results saved to {csv_file}")
