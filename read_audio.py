# %% [markdown]
# # Process Raw Audio Files
# 
# 

# %%
# Start loop through audio file
# Load Audio file
# Set audio start time based on audio file name
# Create a time axis for the audio file based on start time, duration and samplerate
# Align time axis with audio data
# If value in csv is within audio file time range, process it
    # If yes, extract audio clip with start time from csv with duration
# Else skip to next audio file

# %%
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import sys
import logging

def init_logging():

    # Initialize logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='logfile.log', encoding='utf-8', 
                        level=logging.DEBUG, 
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

    return logger

logger = init_logging()
logger.info("********Started processing audio files.*************")

# %%

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

# %%
# Read CSV file with timestamps and durations
file_path = Path.cwd() / 'csv'
logging.info(f"Reading CSV files from directory: {file_path}")

#read all csv files in directory
csv_files = [f for f in file_path.glob('*.csv')]
logging.info(f"Found {len(csv_files)} CSV files.")

# %%
# Read csv file and display head

for file in csv_files:
    df = pd.read_csv(file)

df.head()




# %%
id = df['ID']
date = df['Date']
time_UTC = df['Time (ThinkPad DS) [UTC]']

# Convert date and time to datetime object with format YYYY-MM-DD HH:MM:SS
date_time_UTC = pd.to_datetime(date + ' ' + time_UTC, format='%Y-%m-%d %H:%M:%S')
date_time_UTC.head()

# %%
# Load all audio files
audio_path = Path.cwd() / 'input'
file_types = '*.WAV'  # the tuple of file types
audio_files = sorted([f for f in audio_path.glob(file_types)])

# HACK
audio_file = audio_files[0]
audio_file.stem

# %%
# Read audio file
samplerate, data = wav.read(audio_file)
data.shape, samplerate

memory_size = data.size * data.itemsize
memory_size/1e6 # Unit: MB

# %%
# Determine audio recording duration
duration = 1e9 * data.shape[0] / samplerate # units: seconds [ns]

# %%
# Read start time from audio file name:
start_time = get_start_time_from_filename(audio_file)
start_time

# %%
int(start_time.timestamp() * 1e9)  # units: nanoseconds

# %%
# Generate recording time axis
rec_times_axis = np.linspace(0, duration, len(data))
rec_times_axis

# %%

rts = rec_times_axis.astype('timedelta64[ns]') + pd.to_datetime(start_time)
#rt = pd.to_timedelta(rec_time_axis, unit='ns') + pd.to_datetime(get_start_time_from_filename(audio_file))
rts

# %%
rts.size * rts.itemsize / 1e6  # Unit: MB

# %%
arr_int = rts.astype('int64')
arr_int

# %%
arr_int.size * arr_int.itemsize / 1e6  # Unit: MB

# %%
data.dtype

# %%
step = 100 # or pick dynamically: len(data)//1_000_000 for ~1M points
idx = np.arange(0, len(data), step)

plt.plot(
    pd.to_datetime(start_time) + rec_times_axis[idx].astype('timedelta64[ns]'),
    data[idx]
)
plt.show()

# %%
rts

# %%
# If rts is in nanoseconds (int64)
rts_ns = np.sort(np.array(rts, dtype=np.int64))

# Convert pandas datetime to int64 nanoseconds
dt_ns = date_time_UTC.values.astype('int64')

# %%
rts_ns.dtype

# %%
idxs = np.searchsorted(rts_ns, dt_ns)
idxs

# %%
rts_min = rts_ns.min()
rts_max = rts_ns.max()

for dt in dt_ns:
    mask = (dt >= rts_min) & (dt <= rts_max)
    print(mask)


# %%
len(dt_ns)

# %%
# Downsample for plotting clarity
step = 100
idx = np.arange(0, len(data), step)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    pd.to_datetime(start_time) + rec_times_axis[idx].astype('timedelta64[ns]'),
    data[idx],
    label="Audio waveform"
)

# Convert CSV timestamps to datetime for plotting
for t in date_time_UTC:
    ax.axvline(t, color='red', linestyle='--', linewidth=1, label='CSV timestamp')

# Improve legend (avoid duplicate labels)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.set_title(f"Audio with CSV Timestamps: {audio_file.stem}")
plt.tight_layout()
plt.show()
plt.close() 

# %%
import matplotlib.dates as mdates

clip_duration = 60  # seconds

# Downsample for plotting clarity
step = 100
idx = np.arange(0, len(data), step)
time_axis = pd.to_datetime(start_time) + rec_times_axis[idx].astype('timedelta64[ns]')

# Plot 2: Audio with shaded clip durations (120s after each CSV timestamp)
fig, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(time_axis, data[idx], label="Audio waveform")

for t in date_time_UTC:
    start_t = t
    end_t = t + pd.Timedelta(seconds=clip_duration)
    
    # Only shade region that lies within audio file duration
    if start_t > time_axis[-1] or end_t < time_axis[0]:
        continue  # skip if outside recording
    start_t = max(start_t, time_axis[0])
    end_t = min(end_t, time_axis[-1])
    
    ax2.axvspan(start_t, end_t, color='red', alpha=0.2, label='Clip window')

# Avoid duplicate legend entries
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())

ax2.set_xlabel("Time")
ax2.set_ylabel("Amplitude")
ax2.set_title(f"Audio with Highlighted 120s Clip Windows: {audio_file.stem}")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
plt.tight_layout()
plt.show()
plt.close()


# %%
import matplotlib.dates as mdates

# Parameters
clip_duration = 120   # seconds
clip_delay = 30       # seconds after CSV time before clip starts

# Downsample for plotting clarity
step = 100
idx = np.arange(0, len(data), step)
time_axis = pd.to_datetime(start_time) + rec_times_axis[idx].astype('timedelta64[ns]')

# Plot: Audio with vertical lines and delayed shaded clip windows
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(time_axis, data[idx], label="Audio waveform")

for t in date_time_UTC:
    # --- Vertical red line at CSV timestamp ---
    ax.axvline(t, color='red', linestyle='--', linewidth=1, label='CSV timestamp')

    # --- Shaded region 30s after timestamp lasting 120s ---
    start_t = t + pd.Timedelta(seconds=clip_delay)
    end_t = start_t + pd.Timedelta(seconds=clip_duration)

    # Only shade region that lies within audio file duration
    if start_t > time_axis[-1] or end_t < time_axis[0]:
        continue  # skip if outside recording
    start_t = max(start_t, time_axis[0])
    end_t = min(end_t, time_axis[-1])

    ax.axvspan(start_t, end_t, color='red', alpha=0.2, label=f'Clip window (+{clip_delay}s)')

# Avoid duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())

# Axis formatting
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.set_title(f"Audio with CSV Timestamps and Delayed {clip_duration}s Clip Windows (+{clip_delay}s): {audio_file.stem}")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

plt.tight_layout()
plt.show()


# %%



