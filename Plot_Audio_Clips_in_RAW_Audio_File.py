# # RAW Audio file with CSV Timestamps and Audio Clip Selection

# %%
# Parameters
clip_duration = 60   # seconds
clip_delay = 30       # seconds after CSV time before clip starts

custom_audio_path = '/home/dorian/DTUMaster/data/raw/2025-10_OsloWaveTank/AuralM3'
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

def get_start_time_from_filename(file_path: Path, 
                                 dt_format: str="%Y%m%d_%H%M%S") -> datetime:
    """
    Extracts the start time from the audio filename.
    Parameters:
    file_path (Path): The path to the audio file.
    dt_format (str): The datetime format used in the filename.
    Returns:
    datetime: The extracted start time as a datetime object.
    Raises:
    ValueError: If the filename does not match the expected format.
    """

    try:
        # Remove suffix and parse datetime from the stem
        dt = datetime.strptime(file_path.stem, dt_format)
    except ValueError:
        raise ValueError(
            f"Filename '{file_path.name}' does not match expected format 'YYYYMMDD_HHMMSS.WAV'."
        )
    return dt

def generate_time_axis(start_time: datetime, 
                       data: np.ndarray, samplerate: int) -> np.ndarray:
    """
    Generates a time axis for the audio data.
    Parameters:
    start_time (datetime): The start time of the audio recording.
    data (np.ndarray): The audio data array.
    samplerate (int): The sample rate of the audio data.
    Returns:
    np.ndarray: An array of datetime64 objects representing the time axis.
    """
    n_samples = data.shape[0]
    start_time = np.datetime64(start_time)
    time_axis = start_time + np.arange(n_samples) / samplerate * np.timedelta64(1, 's')

    # Test to make sure shapes match
    assert data.shape == time_axis.shape

    return time_axis

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
            min_val = np.min(d)
            max_val = np.max(d)
            
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
                    return False
                
                elif equal_bounds:
                    logger.warning(f"Audio data reaches boundary limits - possible clipping detected!")
                    logger.warning(f"Audio data min/max values: {min_val}/{max_val}")

                    if min_val == valid_min:
                        logger.warning(f"Minimum value {min_val} equals lower bound {valid_min}")
                    if max_val == valid_max:
                        logger.warning(f"Maximum value {max_val} equals upper bound {valid_max}")
                else:
                    logger.info(f"Audio data is within valid bounds.")
                    return True
            else:
                logger.warning(f"Unknown audio data type: {dtype}")
                return False

# %%

memory_list = []
file_name_list = []
duration_list = []

#%%
# Read CSV file with timestamps and durations


# Validate if custom paths are provided, else use defaults
csv_path = Path(custom_csv_path or Path.cwd() / 'csv')
logger.info(f"Using CSV path: {csv_path}")

audio_path = Path(custom_audio_path or Path.cwd() / 'input')
logger.info(f"Using audio path: {audio_path}")

# Load all CSV files
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
        _ = check_audio_bounds(data)
        
        _, m = get_memory_size_of_np_array(data, threshold=2e9)
        
        memory_list.append(m)

        # Determine audio recording duration
        duration = 1e9 * data.shape[0] / samplerate # units: seconds [ns]
        duration_list.append(duration)

        # Load Stat time from filename
        start_time = get_start_time_from_filename(audio_file)

        # Generate recording time axis
        rec_times_axis = np.linspace(0, duration, len(data))

        # %%
        # Downsample for plotting clarity
        step = 1000 # plot every 100th sample, decreases plotting time and file size
        idx = np.arange(0, len(data), step)
        time_axis = pd.to_datetime(start_time) + rec_times_axis[idx].astype('timedelta64[ns]')

        del rec_times_axis  # free memory

        # Determine audio start and end times
        audio_start = time_axis[0]
        audio_end   = time_axis[-1]

        # Plot: only within audio range
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(time_axis, data[idx], label="Audio waveform", color='black')

        for t, event_id in zip(date_time_UTC, df['ID']):
            # Skip if timestamp is completely outside audio file
            if t < audio_start or t > audio_end:
                continue

            # --- Red dashed vertical line at CSV timestamp ---
            ax.axvline(t, color='red', linestyle='--', linewidth=1, label='CSV Timestamp')

            # --- Annotate the ID near the top of the plot ---
            annotation_label = str(f"ID: {event_id} at {t.strftime('%H:%M:%S')}")
            ax.text(
                t,                      # x position (time)
                ax.get_ylim()[1] * -0.85, # y position (90% up the y-axis)
                annotation_label,          # the ID label
                rotation=90,            # vertical text
                color='red',
                fontsize=9,
                va='bottom',
                ha='right',
                backgroundcolor='none',
                alpha=1
            )

            # --- Shaded region (delayed clip window) ---
            start_t = t + pd.Timedelta(seconds=clip_delay)
            end_t = start_t + pd.Timedelta(seconds=clip_duration)

            if start_t > audio_end or end_t < audio_start:
                continue
            start_t = max(start_t, audio_start)
            end_t = min(end_t, audio_end)

            ax.axvspan(start_t, end_t, color='red', alpha=0.2, label=f'Clip window')
            annotation_label = str(f"Start: {start_t.strftime('%H:%M:%S')}\nDelay: {clip_delay}s\nDuration: {clip_duration}s")
            ax.text(
                start_t,                      # x position (time)
                ax.get_ylim()[1] * 1.01, # y position (90% up the y-axis)
                annotation_label,          # the ID label
                rotation=0,            # vertical text
                color='black',
                fontsize=7,
                va='bottom',
                ha='left',
                backgroundcolor='none',
                alpha=1
            )


        # Restrict x-axis strictly to audio range
        ax.set_xlim(audio_start, audio_end)

        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())

        # Annotation: Audio filename
        ax.text(
            0.99, 0.01,                      # (x, y) in axes fraction coordinates
            audio_file.name,                 # the text
            transform=ax.transAxes,          # use axes-relative coordinates
            fontsize=9,
            color='gray',
            ha='right',                      # horizontal alignment
            va='bottom',                     # vertical alignment
            alpha=0.7,
        )

        # Formatting
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Amplitude RAW Wave Units [int16]")
        ax.set_title(f"RAW Audio file with CSV Timestamps and Audio Clip Selection.", y=1.08)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        plt.tight_layout()
        plt.savefig(f"output/{audio_file.stem}_Audio_Clip_Selection.png", 
                    dpi=72, bbox_inches='tight')
        #plt.close()

        # Clean up large variables to free memory
        legend = ax.get_legend()
        if legend:
            legend.remove()




        fig.clear()
        
        plt.close(fig)
        plt.clf()
        plt.close('all')

        del data, time_axis, fig, ax
        import gc; gc.collect()


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
