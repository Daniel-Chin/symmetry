import pretty_midi as pm
from music21 import converter
from music21.instrument import Instrument, Piano
from midi2audio import FluidSynth
import librosa
import numpy as np

from dataset_config import *

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0]
# PITCH_SEQ_RAISING_12 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

BEND_RATIO = .5  # When MIDI specification is incomplete...
BEND_MAX = 8191
GRACE_END = .1
TEMP_MIDI_FILE = './temp/temp.mid'
TEMP_WAV_FILE  = './temp/temp.wav'

fs = FluidSynth(SOUND_FONT_PATH, sample_rate=SR)

def synthOneNote(
    fs: FluidSynth, pitch: float, instrument: Instrument, 
    temp_wav_file=TEMP_WAV_FILE, verbose=False, 
):
    # make piano midi
    music = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program(
        'Acoustic Grand Piano', 
    )
    piano = pm.Instrument(program=piano_program)
    rounded_pitch = round(pitch)
    note = pm.Note(
        velocity=100, pitch=rounded_pitch, 
        start=0, end=SEC_PER_NOTE + GRACE_END, 
    )
    pitchBend = pm.PitchBend(
        round((pitch - rounded_pitch) * BEND_MAX * BEND_RATIO), 
        time=0, 
    )
    if verbose:
        print(rounded_pitch, ',', pitchBend.pitch)
    piano.notes.append(note)
    piano.pitch_bends.append(pitchBend)
    music.instruments.append(piano)
    music.write(TEMP_MIDI_FILE)

    # turn piano into `intrument`
    s = converter.parse(TEMP_MIDI_FILE)
    for el in s.recurse():
        if 'Instrument' in el.classes:  # or 'Piano'
            el.activeSite.replace(el, instrument)
    s.write('midi', TEMP_MIDI_FILE)

    # synthesize to wav
    fs.midi_to_audio(TEMP_MIDI_FILE, temp_wav_file)

    # read wav
    audio, sr = librosa.load(temp_wav_file, SR)
    assert sr == SR
    return audio

def vibrato():
    music = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program(
        'Acoustic Grand Piano', 
    )
    piano = pm.Instrument(program=piano_program)
    END = 6
    note = pm.Note(
        velocity=100, pitch=60, 
        start=0, end=END, 
    )
    for t in np.linspace(0, END, 100):
        pB = pm.PitchBend(round(
            np.sin(t * 2) * BEND_MAX
        ), time=t)
        piano.pitch_bends.append(pB)
    piano.notes.append(note)
    music.instruments.append(piano)
    music.write(TEMP_MIDI_FILE)

    # synthesize to wav
    fs.midi_to_audio(TEMP_MIDI_FILE, 'vibrato.wav')

def main():
    ...

def testPitchBend():
    # Midi doc does not specify the semantics of pitchbend.  
    # Synthesizers may have inconsistent behaviors. Test!  
    import numpy as np

    for pb in np.linspace(0, 1, 8):
        p = 60 + pb
        synthOneNote(fs, p, Piano(), f'''./temp/{
            format(p, ".2f")
        }.wav''', True)

# main()
testPitchBend()
# vibrato()
