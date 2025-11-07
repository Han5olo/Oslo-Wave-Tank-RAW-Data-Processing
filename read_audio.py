# %% [markdown]
# # Process Raw Audio Files

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
import matplotlib.pyplot as plt
import matplotlib


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

#read all csv files in directory
csv_files = [f for f in file_path.glob('*.csv')]
csv_files


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
audio_files = [f for f in audio_path.glob(file_types)]
audio_files

# %%
audio_file = audio_files[0]

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
# Generate recording time axis
rec_time_axis = np.linspace(0, duration, len(data))
rec_time_axis

# %%

rt = rec_time_axis.astype('timedelta64[ns]') + pd.to_datetime(start_time)
#rt = pd.to_timedelta(rec_time_axis, unit='ns') + pd.to_datetime(get_start_time_from_filename(audio_file))
rt

# %%
rt.size * rt.itemsize / 1e6  # Unit: MB

# %%
data.dtype

# %%

# %%
step = 100 # or pick dynamically: len(data)//1_000_000 for ~1M points
idx = np.arange(0, len(data), step)

plt.plot(
    pd.to_datetime(start_time) + rec_time_axis[idx].astype('timedelta64[ns]'),
    data[idx]
)
plt.show()

# %%



