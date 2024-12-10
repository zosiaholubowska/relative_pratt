import time
import pandas
import slab
import random
import freefield
import os
import pickle
from utils import shuffle_pairs, notetofreq, create_sound
from datetime import datetime
import re

# ========== DIRS AND PARAMS
def load_parameters(subject):
    DIR = os.getcwd()
    STIM_DIR = f'{DIR}/stimuli'
    RESULTS_DIR = f'{DIR}/Results'
    PLOT_DIR = f'{DIR}/plots'
    samplerate = 44828
    table = slab.ResultsTable(subject=subject,
                              columns='timestamp, subject, condition, cond_index, idx, midi_note, frequency, direction, interval, azimuth, elevation')

    return DIR, STIM_DIR, RESULTS_DIR, PLOT_DIR, samplerate, table

# =========== SET-UP
def load_processors(DIR):
    proc_list = [['RX81', 'RX8', f'{DIR}/rcx/sound_test.rcx'],
                 ['RX82', 'RX8', f'{DIR}/rcx/sound_test.rcx'],
                 ['RP2', 'RP2', f'{DIR}/rcx/button_sound.rcx']]

    freefield.initialize('dome', device=proc_list, sensor_tracking=True) #
    #freefield.set_logger('debug')
    directions = [21, 22, 23, 24, 25]

    return proc_list, directions

# ========== STIMULI
def load_tones(STIM_DIR):
    step = 90

    conditions = ['complex', 'viola', 'flute']
    random.shuffle(conditions)

    with open(f'{STIM_DIR}/shuffled_pairs.pickle', 'rb') as f:
        shuffled_pairs = pickle.load(f)

    return step, shuffled_pairs, conditions

### ========== ABSOLUTE MEASURES
def run_pratt(subject, shuffled_pairs, proc_list, table, step, condition, STIM_DIR, cond_index):
    stims = shuffled_pairs[condition]

    print('#################\n## CALIBRATION ## \n#################')
    freefield.calibrate_sensor(led_feedback=True, button_control='processor')  # sensor calibration
    print('#####################\nCALIBRATION COMPLETED\n#####################')

    input("Do you want to continue? (PRESS ENTER): ")
    print("Continuing...")

    stims_length = len(stims)
    #stims_length = len(stims[0:20])

    # ITERATE OVER ALL STIMULI
    for idx, stim in enumerate(stims): #[0:20]
        print(f'STIMULUS: {idx+1} / {stims_length}')
        midi_note = stim[0]
        frequency = notetofreq(midi_note)
        duration = 1.0
        direction = stim[1]
        print(f'Loudspeaker| {direction}\nMIDI note| {midi_note}')
        # === CREATE THE SOUND
        sound = create_sound(frequency=frequency, midi_note=midi_note, duration=duration, condition=condition, STIM_DIR=STIM_DIR)
        sound.level = 80 if condition != "complex" else 70

        # === WRITE THE LOUDSPEAKER ON THE PROCESSOR
        [curr_speaker] = freefield.pick_speakers(direction)
        [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
        freefield.write('data', sound.data, ['RX81', 'RX82'])
        freefield.write('playbuflen', sound.n_samples, ['RX81', 'RX82'])
        freefield.write('channel', curr_speaker.analog_channel, curr_speaker.analog_proc)
        freefield.write('channel', 99, other_proc)
        freefield.play()
        time.sleep(2.0)
        # ===== PREPARE THE TONE FOR BUTTON CONFIRMATION
        tone = slab.Sound.pinknoise(duration=0.1)
        [curr_speaker] = freefield.pick_speakers(23)
        [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
        freefield.write('data', tone.data, curr_speaker.analog_proc)
        freefield.write('playbuflen', tone.n_samples, ['RX81', 'RX82'])
        freefield.write('channel', curr_speaker.analog_channel, curr_speaker.analog_proc)
        freefield.write('channel', 99, other_proc)
        freefield.write(tag='bitmask', value=8, processors='RX81')  # illuminate LED
        response = 0
        while not response:
            response = freefield.read('response', processor='RP2')
        pose = freefield.get_head_pose(method='sensor')  # read the head position
        if all(pose):
            print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
        else:
            print('no head pose detected', end="\r", flush=True)
        if all(pose):
            print('Response| azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
        freefield.write('chan', 99, ['RX81', 'RX82'])
        freefield.write(tag='bitmask', value=0, processors='RX81')  # turn the light off LED
        freefield.play()
        response = freefield.read('response', 'RP2', 0)
        row = table.Row(timestamp=datetime.now(), subject = subject, condition = condition, cond_index = cond_index,
                        idx = idx, midi_note = midi_note, frequency = frequency, direction = direction, interval= 'NA',
                        azimuth = pose[0], elevation = pose[1])
        table.write(row)
        time.sleep(1)
        if (idx+1) == stims_length:
            print('End of the first task.')
            print('Take a break')

        elif (idx+1) % step == 0:
            print('Time to take a break')
            input("Do you want to CALIBRATE? (PRESS ENTER): ")

            print('##########\nCALIBRATION\n##########')
            freefield.calibrate_sensor(led_feedback=True, button_control='processor')  # sensor calibration
            print('##########\nCALIBRATION COMPLETED\n##########')

            input("Do you want to continue? (PRESS ENTER): ")
            print("Continuing...")



### ========== RELATIVE MEASURES

# ========== STIMULI

def load_stimuli_relative(STIM_DIR, directions):

    files = [f for f in os.listdir(STIM_DIR) if 'stim_' in f]
    random.shuffle(files)

    pairs = [(sound, speaker) for sound in files for speaker in directions]
    stims = shuffle_pairs(pairs)
    tone = slab.Sound.pinknoise(duration=0.1)  # tone to confirm button press

    step = 50

    return step, stims, tone


def run_rel(subject, stims, STIM_DIR, samplerate, proc_list, step, table):

    print('#################\n## CALIBRATION ## \n#################')
    freefield.calibrate_sensor(led_feedback=True, button_control=True)  # sensor calibration
    print('#####################\nCALIBRATION COMPLETED\n#####################')

    input("Do you want to continue? (PRESS ENTER): ")
    print("Continuing...")

    stims_length = len(stims)
    task = 'relative'

    for idx, stim in enumerate(stims):

        task = 'absolute'
        match = re.search(r'_(\-?\d+\.?\d*)\.csv$', stim[0])
        if match:
            interval = match.group(1)

        print(f'STIMULUS: {idx + 1} / {stims_length}')
        file_name = stim[0]
        stim_params = pandas.read_csv(f'{STIM_DIR}/{file_name}')
        times = stim_params['onset_sec'].tolist()
        time_0 = times[0]
        times = [x - time_0 for x in times]
        frequencies = stim_params['freq'].tolist()
        durations = stim_params['duration'].tolist()
        [curr_speaker] = freefield.pick_speakers(stim[1])
        start_time = time.time()
        i=0

        while time.time() - start_time < times[-1]:

            if time.time() - start_time > times[i]:
                freefield.write('f0', frequencies[i], ['RX81', 'RX82'])

                duration = durations[i]  # duration in seconds
                freefield.write('len', int(duration * samplerate * 0.95), ['RX81', 'RX82'])

                freefield.write('chan', curr_speaker.analog_channel, curr_speaker.analog_proc)
                [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
                freefield.write('chan', 99, other_proc)

                freefield.play()
                print(time.time() - start_time)
                i += 1

        time.sleep(duration)

        freefield.write(tag='bitmask', value=8, processors='RX81')  # illuminate LED
        response = 0
        while not response:
            pose = freefield.get_head_pose(method='sensor')
            if all(pose):
                print('head pose: azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]), end="\r", flush=True)
            else:
                print('no head pose detected', end="\r", flush=True)
            response = freefield.read('response', processor='RP2')
        if all(pose):
            print('Response| azimuth: %.1f, elevation: %.1f' % (pose[0], pose[1]))
        freefield.write('chan', 99, ['RX81', 'RX82'])
        freefield.write(tag='channel', value=1, processors='RX81')  # illuminate LED
        freefield.write(tag='bitmask', value=0, processors='RX81')  # illuminate LED
        freefield.play()
        response = freefield.read('response', 'RP2', 0)

        time.sleep(1)
        row = table.Row(timestamp=datetime.now(), subject=subject, task=task,
                        idx=idx, stim=stim, frequency=frequencies[i], direction=stim[1], interval = interval,
                        azimuth=pose[0], elevation=pose[1])
        table.write(row)

        if (idx+1) == stims_length:
            print('End of the second task.')
            print('THANK YOU!')

        elif (idx+1) % step == 0:
            print('Time to take a break')
            input("Do you want to CALIBRATE? (PRESS ENTER): ")

            print('##########\nCALIBRATION\n##########')
            freefield.calibrate_sensor(led_feedback=True, button_control=True)  # sensor calibration
            print('##########\nCALIBRATION COMPLETED\n##########')

            input("Do you want to continue? (PRESS ENTER): ")
            print("Continuing...")



