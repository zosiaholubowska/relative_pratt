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

subject = 'test_run'

# DIRECTORIES
DIR = os.getcwd()
RESULTS_DIR = f'{DIR}/Results'
samplerate = 44828

table = slab.ResultsTable(subject=subject,
                              columns='timestamp, subject, direction, azimuth, elevation')


# SET-UP
proc_list = [['RX81', 'RX8', f'{DIR}/rcx/sound_test.rcx'],
                 ['RX82', 'RX8', f'{DIR}/rcx/sound_test.rcx'],
                 ['RP2', 'RP2', f'{DIR}/rcx/button_sound.rcx']]

freefield.initialize('dome', device=proc_list, sensor_tracking=True) #
#freefield.set_logger('debug')
directions = [21, 22, 23, 24, 25]

# SOUND and SEQUENCE

tone = slab.Sound.pinknoise(duration=1)

#todo: create a sequence of 20 repetitions for each directions

# CALIBRATION

freefield.calibrate_sensor(led_feedback=True, button_control=True)

# TEST

tone.level = 80

for direction in sequence:
    [curr_speaker] = freefield.pick_speakers(direction)
    [other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
    freefield.write('data', tone.data, ['RX81', 'RX82'])
    freefield.write('playbuflen', tone.n_samples, ['RX81', 'RX82'])
    freefield.write('channel', curr_speaker.analog_channel, curr_speaker.analog_proc)
    freefield.write('channel', 99, other_proc)
    freefield.play()

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

    response = freefield.read('response', 'RP2', 0)
    row = table.Row(timestamp=datetime.now(), subject=subject,
                    direction=direction,
                    azimuth=pose[0], elevation=pose[1])
    table.write(row)
    time.sleep(1)

print('TEST COMPLETED.')

