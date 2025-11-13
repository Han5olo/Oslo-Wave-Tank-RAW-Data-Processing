

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
logger = logging.getLogger(__name__)

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
            


def calc_end_time(start_time: datetime, duration_ns: float) -> datetime:
    """Calculate end time given start time and duration in nanoseconds.
    Parameters:
    start_time (datetime): The start time of the audio recording.
    duration_ns (float): The duration of the audio recording in nanoseconds.
    Returns:
    datetime: The calculated end time. 
    """
    duration = timedelta(microseconds=duration_ns / 1e3)  # convert ns to us
    end_time = start_time + duration
    
    return end_time 


def get_time_range_mask(date_time_UTC: pd.Series, 
                          start_time: datetime, 
                          end_time: datetime) -> pd.Series:
    """
    Create a boolean mask for timestamps within a specified time range.
    Parameters:
    date_time_UTC (pd.Series): Series of datetime objects.
    start_time (datetime): Start time of the range.
    end_time (datetime): End time of the range.
    Returns:
    pd.Series: Boolean mask indicating which timestamps are within the range.
    """
    mask = (date_time_UTC >= start_time) & (date_time_UTC <= end_time)
    return mask

def create_audio_clip(samplerate, data, rec_time, id, run, start_time, 
                      clip_delay, clip_duration):
    """Creates an audio clip from the raw audio data given the recording time and parameters.
    Parameters:
    samplerate (int): The sample rate of the audio data.
    data (np.ndarray): The audio data array.
    rec_time (datetime): The recording time for the clip.
    id (str): The ID associated with the recording.
    run (str): The run associated with the recording.
    start_time (datetime): The start time of the audio recording.
    clip_delay (int): The delay in seconds after the recording time before the clip starts.
    clip_duration (int): The duration of the audio clip in seconds.
    Returns:
    None 
    """
    
    start = (rec_time - start_time).total_seconds() + clip_delay
    end = start + clip_duration

    logger.info(f"Creating clip {id}: {rec_time} -> {start:.1f}s to {end:.1f}s")
    
    start_sample = int(start * samplerate)
    end_sample = int(end * samplerate)
    segment = data[start_sample:end_sample]

    output_filename = f"output/audio_clips/{id}_{run}_{rec_time.strftime('%Y%m%d_%H%M%S')}.wav"
    try:
        wav.write(output_filename, samplerate, segment)
        logger.info(f"> Saved audio clip to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save audio clip {output_filename}: {e}")
    