import os
from os import path
import shutil
import pickle

import pretty_midi as pm
from music21.instrument import Instrument, Piano
try:
    # my fork
    from midi2audio_fork.midi2audio import FluidSynth
except ImportError:
    # fallback
    from midi2audio import FluidSynth
import librosa
import soundfile
import numpy as np
from tqdm import tqdm

from dataset_config import *
from intruments_and_ranges import intruments_ranges

SONG_LEN = (
    N_SAMPLES_PER_NOTE + N_SAMPLES_BETWEEN_NOTES
) * len(MAJOR_SCALE)

BEND_RATIO = .5  # When MIDI specification is incomplete...
BEND_MAX = 8191
GRACE_END = .1
TEMP_MIDI_FILE = './temp/temp.mid'
TEMP_WAV_FILE  = './temp/temp.wav'
DATASET_PATH = './datasets/cleanTrain'

fs = FluidSynth(SOUND_FONT_PATH, sample_rate=SR)

def synthOneNote(
    fs: FluidSynth, pitch: float, instrument: Instrument, 
    temp_wav_file=TEMP_WAV_FILE, verbose=False, 
):
    # make midi
    music = pm.PrettyMIDI()
    ins = pm.Instrument(program=instrument.midiProgram)
    rounded_pitch = int(round(pitch))
    note = pm.Note(
        velocity=100, pitch=rounded_pitch, 
        start=0, end=NOTE_DURATION + GRACE_END, 
    )
    pitchBend = pm.PitchBend(
        round((pitch - rounded_pitch) * BEND_MAX * BEND_RATIO), 
        time=0, 
    )
    if verbose:
        print(rounded_pitch, ',', pitchBend.pitch)
    ins.notes.append(note)
    ins.pitch_bends.append(pitchBend)
    music.instruments.append(ins)
    music.write(TEMP_MIDI_FILE)

    # synthesize to wav
    fs.midi_to_audio(TEMP_MIDI_FILE, temp_wav_file, verbose=False)

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
            np.sin(t * 5) * BEND_MAX
        ), time=t)
        piano.pitch_bends.append(pB)
    piano.notes.append(note)
    music.instruments.append(piano)
    music.write(TEMP_MIDI_FILE)

    # synthesize to wav
    fs.midi_to_audio(TEMP_MIDI_FILE, 'vibrato.wav')

def testPitchBend():
    # Midi doc does not specify the semantics of pitchbend.  
    # Synthesizers may have inconsistent behaviors. Test!  

    for pb in np.linspace(0, 1, 8):
        p = 60 + pb
        synthOneNote(fs, p, Piano(), f'''./temp/{
            format(p, ".2f")
        }.wav''', True)

def main():
    try:
        shutil.rmtree(DATASET_PATH)
    except FileNotFoundError:
        pass
    os.makedirs(DATASET_PATH, exist_ok=True)
    index = []
    for instrument, pitch_range in tqdm(intruments_ranges):
        pitches_audio = {}
        for pitch in pitch_range:
            audio = synthOneNote(
                fs, pitch, instrument, 
            )[:N_SAMPLES_PER_NOTE]
            pitches_audio[pitch] = audio
            dtype = audio.dtype
            audio[-FADE_OUT_N_SAMPLES:] = audio[
                -FADE_OUT_N_SAMPLES:
            ] * FADE_OUT_FILTER
        for start_pitch, song in GenSongs(
            pitch_range, pitches_audio, dtype, 
        ):
            index.append((instrument.instrumentName, start_pitch))
            soundfile.write(path.join(
                DATASET_PATH, 
                f'{instrument.instrumentName}-{start_pitch}.wav', 
            ), song, SR)
    with open(path.join(DATASET_PATH, 'index.pickle'), 'wb') as f:
        pickle.dump(index, f)

def GenSongs(pitch_range: range, pitches_audio, dtype):
    start_pitch = pitch_range.start
    while True:
        song = np.zeros((SONG_LEN, ), dtype=dtype)
        cursor = N_SAMPLES_BETWEEN_NOTES
        for d_pitch in MAJOR_SCALE:
            pitch = start_pitch + d_pitch
            try:
                audio = pitches_audio[pitch]
            except KeyError:
                return
            song[
                cursor : cursor + N_SAMPLES_PER_NOTE
            ] = audio
            cursor += N_SAMPLES_PER_NOTE
            cursor += N_SAMPLES_BETWEEN_NOTES
        assert cursor - N_SAMPLES_BETWEEN_NOTES == SONG_LEN
        yield start_pitch, song
        start_pitch += 1

# vibrato()
# testPitchBend()
main()
