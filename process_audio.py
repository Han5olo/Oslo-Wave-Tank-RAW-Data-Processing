# %%
# Parameters
clip_duration = 10   # seconds
clip_delay = 25       # seconds after CSV time before clip starts

custom_audio_path = None
custom_csv_path = '/home/dorian/Desktop/OsloCSV/rec'
custom_audio_path = '/home/dorian/DTUMaster/data/raw/2025-10_OsloWaveTank/AudioMoth'

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import os
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import wave
import func

#Get logger
logger = func.init_logging()

#%% Locate CSV file
csv_path = Path(custom_csv_path or Path.cwd() / 'csv')
csv_files = sorted([f for f in csv_path.glob('*.csv')])

for csv_file in csv_files:
    logger.info(f"Found CSV file: {csv_file}")

    #%% Read CSV file as pandas DataFrame
    df = pd.read_csv(csv_file)

    #ids = df['ID']
    #runs = df['Run']
    date = df['Date']
    time_UTC = df['Time (ThinkPad DS) [UTC]']

    # Convert data and time to datetime object with format YYYY-MM-DD HH:MM:SS
    date_time_UTC = pd.to_datetime(date + ' ' + time_UTC, format='%Y-%m-%d %H:%M:%S')

    #%% Locate audio files
    audio_path = Path(custom_audio_path or Path.cwd() / 'input')
    audio_files = sorted([f for f in audio_path.rglob('*.WAV')])
    logger.info(f"Found {len(audio_files)} audio files in {audio_path}")

    for audio_file in audio_files:

        logger.info(f"Processing audio file: {audio_file}")
        # Load Stat time from filename
        start_time = func.get_start_time_from_filename(audio_file)

        #%% Read audio file duration using wave module
        logger.info("Reading audio file duration...")
        with wave.open(str(audio_file), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_seconds = frames / float(rate)
            duration = int(duration_seconds * 1e9) # To nanoseconds

        #%%

        end_time = func.calc_end_time(start_time, duration)

        #%% Find recorded timestamps within audio file time range

        # Create boolean mask for timestamps within audio file time range
        mask = func.get_time_range_mask(date_time_UTC, start_time, end_time)

        #%% Create audio clips for recorded timestamps within audio file time range
        if any(mask):
            logger.info("Recorded timestamps found within audio file time range.")

            # Apply mask to get recorded timestamps within audio file time range

            ids = df['ID'][mask]
            runs = df['Run'][mask]            
            rec_timestamps = date_time_UTC[mask]

            # Read audio file using scipy wav module
            samplerate, data = wav.read(audio_file)

            # Check duration calculation
            assert duration == int(1e9 * data.shape[0] / samplerate) # units: seconds [ns]

            duration = 1e9 * data.shape[0] / samplerate

            # Create audio clips based on recorded timestamps
            for (rec_time, id, run) in zip(rec_timestamps, ids, runs):
                logger.info(f"Processing clip for timestamp: {rec_time}, ID: {id}, Run: {run}")
                _ = func.create_audio_clip(samplerate, data, rec_time, id, run,
                                           start_time, clip_delay, 
                                           clip_duration)

            logger.info("Finished creating audio clips.")
            del data, rec_timestamps, ids, runs  # Free up memory
            logger.info("------------------------------------------------------")
        else:
            logger.info("No recorded timestamps within audio file time range.")
            logger.info("------------------------------------------------------")

logger.info("All done!")
logger.info("******************************************************")
# %%
