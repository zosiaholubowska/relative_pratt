import os
import freefield
import slab
import time

DIR = os.getcwd()
STIM_DIR = f'{DIR}/stimuli'
samplerate = 44828

proc_list = [['RX81', 'RX8', f'{DIR}/rcx/sound_test.rcx'],
             ['RX82', 'RX8', f'{DIR}/rcx/sound_test.rcx'],
             ['RP2', 'RP2', f'{DIR}/rcx/button_sound.rcx']]

freefield.initialize('dome', device=proc_list, sensor_tracking=True) #
freefield.set_logger('debug')
freefield.write(tag='bitmask', value=0, processors='RX81')

frequency = 261
duration = 1.0
direction = 21

[curr_speaker] = freefield.pick_speakers(direction)

stim = slab.Sound.tone(frequency=frequency, duration=duration)
stim.level = 85
[other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
freefield.write('data', stim.data, ['RX81', 'RX82'])
freefield.write('playbuflen', stim.n_samples, ['RX81', 'RX82'])
freefield.write('channel', curr_speaker.analog_channel, curr_speaker.analog_proc)
freefield.write('channel', 99, other_proc)
freefield.play()

tone = slab.Sound.pinknoise(duration=0.1)  # tone to confirm button press
[curr_speaker] = freefield.pick_speakers(23)
[other_proc] = [item for item in [proc_list[0][0], proc_list[1][0]] if item != curr_speaker.analog_proc]
freefield.write('data', tone.data, curr_speaker.analog_proc)
freefield.write('playbuflen', tone.n_samples, ['RX81', 'RX82'])
freefield.write('channel', curr_speaker.analog_channel, curr_speaker.analog_proc)
freefield.write('channel', 99, other_proc)
#time.sleep(8)

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

start_time = time.time()
prev_response = 0
while time.time() - start_time < 10:
    response = freefield.read('response', 'RP2', 0)
    if response > prev_response:
        print('button pressed')
        resp_true = 'YES'

    prev_response = response