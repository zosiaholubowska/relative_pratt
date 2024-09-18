import time
import pandas
import slab
import random
import freefield
import os
import pickle
from utils import shuffle_pairs

# ========== DIRS AND PARAMS
DIR = os.getcwd()
STIM_DIR = f'{DIR}/stimuli'
samplerate = 44828
### ========== ABSOLUTE MEASURES

# =========== SET-UP

proc_list = [['RX81', 'RX8', f'{DIR}/rcx/piano.rcx'],
                ['RX82', 'RX8', f'{DIR}/rcx/piano.rcx'],
                 ['RP2', 'RP2', f'{DIR}/rcx/button.rcx']]

freefield.initialize('dome', device=proc_list, sensor_tracking=True)
freefield.set_logger('warning')
directions = [21, 22, 23, 24, 25]

step = 5

# ========== STIMULI
with open(f'{STIM_DIR}/freqs.pickle', 'rb') as fp:
    range_freq = pickle.load(fp)

pairs = [(sound, speaker) for sound in range_freq for speaker in directions for _ in range(3)]
stims = shuffle_pairs(pairs)
tone = slab.Sound.pinknoise(duration=0.1)
freefield.set_signal_and_speaker(signal=tone, speaker=23, chan_tag='channel')

def run_abs(subject, stims):
    # TO DO RESULTS DF
    print('##########\nCALIBRATION\n##########')
    freefield.calibrate_sensor(led_feedback=True, button_control=True)  # sensor calibration
    print('##########\nCALIBRATION COMPLETED\n##########')

    input("Do you want to continue? (PRESS ENTER): ")
    print("Continuing...")
    stims_length = len(stims)

    for idx, stim in enumerate(stims):
        print(f'STIMULUS: {idx+1} / {stims_length}')
        frequency = stim[0]
        duration = 1
        direction = stim[1]

        [curr_speaker] = freefield.pick_speakers(direction)

        freefield.write('channel', 99, ['RX81', 'RX82'])
        freefield.write('f0', frequency, ['RX81', 'RX82'])
        freefield.write('len', int(duration * samplerate * 0.95), ['RX81', 'RX82'])
        freefield.write('chan', curr_speaker.analog_channel, curr_speaker.analog_proc)
        [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
        freefield.write('chan', 99, other_proc)
        freefield.play()
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

        if (idx+1) % step == 0:
            print('Time to take a break')
            input("Do you want to CALIBRATE? (PRESS ENTER): ")

            print('##########\nCALIBRATION\n##########')
            freefield.calibrate_sensor(led_feedback=True, button_control=True)  # sensor calibration
            print('##########\nCALIBRATION COMPLETED\n##########')

            input("Do you want to continue? (PRESS ENTER): ")
            print("Continuing...")



### ========== INDUCED CONTEXT

# ========== STIMULI

files = [f for f in os.listdir(STIM_DIR) if 'stim_' in f]
random.shuffle(files)

pairs = [(sound, speaker) for sound in files for speaker in directions]
stims = shuffle_pairs(pairs)

stims_length = len(stims)


def run_rel(stims):

    for idx, stim in enumerate(stims):

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



