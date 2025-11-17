# %%
# Parameters
import vars

custom_audio_path = None
custom_csv_path = None
v = vars.get_vars() 

# Defining variabels
custom_csv_path = v['custom_csv_path']
custom_audio_path = v['custom_audio_path']
clip_duration = v['clip_duration']
clip_delay = v['clip_delay']

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from scipy import signal
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

num_clips_created = 0
num_audio_files_processed = 0
no_processed_files = list()

def plot_downsample_data(ydem, duration, fname: str = 'sample_plot', *, ftype: str = 'jpg', dpi: int = 72):
    """ PLot downsampled data and save figure as file.
    inputs: ydem - downsampled data
            duration - duration of the data
    returns: None
    """    
    xdem = np.linspace(0, duration, len(ydem))
    plt.figure(figsize=(8, 4))   
    plt.plot(xdem, ydem)
    #plt.axvspan(xdem[300000], xdem[800000], alpha=0.2, color='orange')
    plt.tight_layout() # removes extra whitespace
    plt.legend(['data'], loc='best')
    plt.savefig(f'output/figs/{fname}.{ftype}', dpi=dpi)
    plt.show()
    plt.close()




#%% Locate CSV file
csv_path = Path(custom_csv_path or Path.cwd() / 'csv')
csv_files = sorted([f for f in csv_path.glob('*.csv')])

for csv_file in csv_files:
    logger.info(f"Found CSV file: {csv_file}")

    # Read CSV file as pandas DataFrame
    df = pd.read_csv(csv_file)

    #ids = df['ID']
    #runs = df['Run']
    date = df['Date']
    time_UTC = df['Time (ThinkPad DS) [UTC]']

    # Convert data and time to datetime object with format YYYY-MM-DD HH:MM:SS
    date_time_UTC = pd.to_datetime(date + ' ' + time_UTC, format='%Y-%m-%d %H:%M:%S')

    # Locate audio files
    audio_path = Path(custom_audio_path or Path.cwd() / 'input')
    audio_files = sorted([f for f in audio_path.rglob('*.WAV')])
    logger.info(f"Found {len(audio_files)} audio files in {audio_path}")

    for audio_file in audio_files:

        logger.info(f"Processing audio file: {audio_file}")
        # Load Stat time from filename
        start_time = func.get_start_time_from_filename(audio_file)

        # Read audio file duration using wave module
        logger.info("Reading audio file duration...")
        with wave.open(str(audio_file), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration_seconds = frames / float(rate)
            duration = int(duration_seconds * 1e9) # To nanoseconds

        end_time = func.calc_end_time(start_time, duration)

        # Find recorded timestamps within audio file time range

        # Create boolean mask for timestamps within audio file time range
        mask = func.get_time_range_mask(date_time_UTC, start_time, end_time)

        # Create audio clips for recorded timestamps within audio file time range
        if any(mask):
            
            num_audio_files_processed += 1
            
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

                fname = f"{id}_{run}_{rec_time.strftime('%Y%m%d_%H%M%S')}" 

                logger.info(f"Processing clip for timestamp: {rec_time}, ID: {id}, Run: {run}")
                #_ = func.create_audio_clip(samplerate, data, rec_time, start_time, clip_delay, clip_duration, fname=fname)
                
                # HECK
                logger.debug('Skip creteion of Audio clip')

                num_clips_created += 1


            logger.info("Finished creating audio clips.")
            #del data, rec_timestamps, ids, runs  # Free up memory
            logger.info("------------------------------------------------------")
        else:
            no_processed_files.append(audio_file)
            logger.info("No recorded timestamps within audio file time range.")
            logger.info("------------------------------------------------------")

# create stat file
func.write_stats(len(audio_files), num_audio_files_processed, no_processed_files, num_clips_created)

logger.info(f"{num_audio_files_processed/len(audio_files)}% audio files processed.")
logger.info(f'Total number of audio clips created: {num_clips_created}')
logger.info("All done!")
logger.info("******************************************************")

#%%
q =100
ydem = signal.decimate(data, q)
plot_downsample_data(ydem, duration)
