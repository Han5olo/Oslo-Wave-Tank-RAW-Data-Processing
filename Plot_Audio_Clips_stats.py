# # RAW Audio file with CSV Timestamps and Audio Clip Selection

# %%
# Parameters
clip_duration = 60   # seconds
clip_delay = 30       # seconds after CSV time before clip starts

custom_audio_path = '/home/dorian/Desktop/AudioMoth/files/2025-10-29_Oslo_Day1'
custom_csv_path = '/home/dorian/Desktop/OsloCSV/rec'


# %%
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import os
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging

import sys


def init_logging(script_name: Path = None) -> logging.Logger:

    # Initialize logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'logfile_{script_name}.log', encoding='utf-8', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', 
                        datefmt='%Y/%m/%d %I:%M:%S')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    logger.info("********************************************************")
    logger.info(f"Starting script: {script_name}")

    return logger

logger = init_logging(Path(__file__).name)

def get_memory_size_of_np_array(np_array: np.ndarray, 
                                    threshold: int = None):
    """
    Calculate memory size of a numpy array and check against threshold.
    Parameters:
    np_array (np.ndarray): The numpy array to evaluate.
    threshold (int, optional): Memory size threshold in bytes. If provided,
                            function returns False if exceeded.
    Returns:
    bool: True if within threshold or no threshold provided, False otherwise.
    """
    memory_size = np_array.size * np_array.itemsize

    logger.info(f"Size of var size: {memory_size} bytes "
                f"= {memory_size * 1e-9:.3f} GB")
    if threshold and memory_size > threshold:
        logger.warning(f"Memory size {memory_size} exceeds "
                       f"threshold of {threshold} bytes.")
        return False, memory_size
    return True, memory_size

def check_audio_bounds(d: np.ndarray) -> bool:
            """
            Check if audio data is within valid bounds for its data type.
            =====================  ===========  ===========  =============
                 WAV format            Min          Max       NumPy dtype
            =====================  ===========  ===========  =============
            32-bit floating-point  -1.0         +1.0         float32
            32-bit integer PCM     -2147483648  +2147483647  int32
            24-bit integer PCM     -2147483648  +2147483392  int32
            16-bit integer PCM     -32768       +32767       int16
            8-bit integer PCM      0            255          uint8
            =====================  ===========  ===========  =============
            
            Parameters:
            d (np.ndarray): The audio data array.
            Returns:
            bool: True if data is within bounds, False if out of bounds.
            Valid ranges for common WAV formats:

            """
        
            dtype = d.dtype
            min_val = np.min(data)
            max_val = np.max(data)
            
            bounds = {
                'float32': (-1.0, 1.0),
                'int32': (-2147483648, 2147483647),
                'int16': (-32768, 32767),
                'uint8': (0, 255)
            }
            
            if dtype.name in bounds:
                valid_min, valid_max = bounds[dtype.name]
                out_of_bounds = (min_val < valid_min) or (max_val > valid_max)
                equal_bounds = (min_val == valid_min) or (max_val == valid_max)
                              
                

                logger.info(f"Audio data type: {dtype}")
                logger.info(f"Data range: {min_val} to {max_val}")
                logger.info(f"Valid range for {dtype}: {valid_min} to {valid_max}")
                
                if out_of_bounds:
                    logger.warning(f"Audio data is OUT OF BOUNDS!")
                    return True, min_val, max_val, valid_min, valid_max
                
                elif equal_bounds:
                    logger.warning(f"Audio data reaches boundary limits - possible clipping detected!")
                    logger.warning(f"Audio data min/max values: {min_val}/{max_val}")

                    if min_val == valid_min:
                        logger.warning(f"Minimum value {min_val} equals lower bound {valid_min}")
                    if max_val == valid_max:
                        logger.warning(f"Maximum value {max_val} equals upper bound {valid_max}")
                    
                    return True, min_val, max_val, valid_min, valid_max
                else:
                    logger.info(f"Audio data is within valid bounds.")
                    return True, min_val, max_val, valid_min, valid_max
            else:
                logger.warning(f"Unknown audio data type: {dtype}")
                return False, min_val, max_val, valid_min, valid_max

# %%

memory_list = []
file_name_list = []
duration_list = []
audio_max_amplitude_list = []
audio_min_amplitude_list = []
bounds_min_list = []
bounds_max_list = []


#%%
# Read CSV file with timestamps and durations

if custom_csv_path:
    csv_path = Path(custom_csv_path)
    logger.info(f"Using custom CSV path: {csv_path}")
else:
    csv_path = Path.cwd() / 'csv'
    logger.info(f"Using CSV path: {csv_path}")

csv_files = [f for f in csv_path.glob('*.csv')]
logger.info(f"Found {len(csv_files)} CSV files for processing.")

# ### Loading CSV File
for csv_file in csv_files:
    logger.info(f"########## Processing CSV file: {csv_file.name}")
    # Rad CSV file
    df = pd.read_csv(csv_file)

    # Extract relevant columns
    id = df['ID']
    date = df['Date']
    time_UTC = df['Time (ThinkPad DS) [UTC]']

    # Convert date and time to datetime object with format YYYY-MM-DD HH:MM:SS
    date_time_UTC = pd.to_datetime(date + ' ' + time_UTC, format='%Y-%m-%d %H:%M:%S')

    # %%
    # Load all audio files
    
    if custom_audio_path: 
        audio_path = Path(custom_audio_path)
        logger.info(f"Using custom audio path: {audio_path}")
    else:
        audio_path = Path.cwd() / 'input'
        logger.info(f"Using audio path: {audio_path}")

    file_types = '*.WAV'  # the tuple of file types
    audio_files = sorted([f for f in audio_path.rglob(file_types)])
    logger.info(f"Found {len(audio_files)} {file_types} files for processing.")

    # ### Load Audio Data
    # %%
    for audio_file in audio_files:
        logger.info(f">>>>>>Processing audio file: {audio_file.name}")
        file_name_list.append(audio_file.name)

        # Read audio file
        samplerate, data = wav.read(audio_file)
        data.shape, samplerate
        
        # Check the audio data
        _, amp_min, amp_max, b_min, b_max = check_audio_bounds(data)
        
        audio_max_amplitude_list.append(amp_max)
        audio_min_amplitude_list.append(amp_min)
        bounds_min_list.append(b_min)
        bounds_max_list.append(b_max)


        _, m = get_memory_size_of_np_array(data, threshold=2e9)
        
        memory_list.append(m)

        # Determine audio recording duration
        duration = 1e9 * data.shape[0] / samplerate # units: seconds [ns]
        duration_list.append(duration)


# %%

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
# Memory size subplot
ax1.bar(file_name_list, np.array(memory_list) * 1e-9)
ax1.set_xticklabels(file_name_list, rotation=90)
ax1.set_ylabel("Memory Size (GB)")
ax1.set_title("Memory Size of Audio Files")

# Duration subplot
ax2.bar(file_name_list, np.array(duration_list) / 3600e9)
ax2.set_xticklabels(file_name_list, rotation=90)
ax2.set_ylabel("Duration (hours)")
ax2.set_title("Duration of Audio Files")

plt.tight_layout()
plt.savefig("output/Audio_Files_Analysis.pdf", dpi=72)  
plt.close()





# Audio amplitude bounds visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

x_pos = np.arange(len(file_name_list))

# Plot bounds in red
ax.plot(x_pos, bounds_min_list, 'r-', linewidth=2, label='Lower Bound', marker='o')
ax.plot(x_pos, bounds_max_list, 'r-', linewidth=2, label='Upper Bound', marker='o')

# Plot actual value range in gray
ax.plot(x_pos, audio_min_amplitude_list, 'gray', linewidth=1, label='Actual Min', marker='s', alpha=0.7)
ax.plot(x_pos, audio_max_amplitude_list, 'gray', linewidth=1, label='Actual Max', marker='s', alpha=0.7)

# Fill between actual min/max to show value range
ax.fill_between(x_pos, audio_min_amplitude_list, audio_max_amplitude_list, 
                alpha=0.3, color='gray', label='Actual Range')

ax.set_xticks(x_pos)
ax.set_xticklabels(file_name_list, rotation=90)
ax.set_ylabel("Amplitude")
ax.set_title("Audio Amplitude Bounds vs Actual Values")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/Audio_Amplitude_Bounds.pdf", dpi=72)
plt.close()