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

from dataset_config import *
from intruments_and_ranges import intruments_ranges

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0]
# PITCH_SEQ_RAISING_12 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

SONG_LEN = N_SAMPLES_PER_NOTE * len(MAJOR_SCALE)

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
    # make midi
    music = pm.PrettyMIDI()
    ins = pm.Instrument(program=instrument.midiProgram)
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
    for instrument, pitch_range in intruments_ranges:
        pitches_audio = {}
        for pitch in pitch_range:
            pitches_audio[pitch] = synthOneNote(fs, pitch, instrument)
            dtype = pitches_audio[pitch].dtype
        song = genOneSong(pitch_range, pitches_audio, dtype)
        soundfile.write(f'./datasets/.wav', song, SR)

def genOneSong(pitch_range: range, pitches_audio, dtype):
    start_pitch = pitch_range.start
    while True:
        song = np.zeros((SONG_LEN, ), dtype=dtype)
        cursor = 0
        for d_pitch in MAJOR_SCALE:
            pitch = start_pitch + d_pitch
            try:
                audio = pitches_audio[pitch]
            except KeyError:
                return song
            song[
                cursor : cursor + N_SAMPLES_PER_NOTE
            ] = audio
            cursor += N_SAMPLES_PER_NOTE
        assert cursor == SONG_LEN
        start_pitch += 1

# vibrato()
# testPitchBend()
main()
