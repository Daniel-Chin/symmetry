import os

from legacyIntruments import LEGACY_INSTRUMENTS

def main():
    pitch_map = {}
    for filename in os.listdir('main'):
        x = filename.split('.wav')[0]
        instrument_name, pitch = x.split('-')
        pitch = int(pitch)
        if instrument_name not in pitch_map:
            pitch_map[instrument_name] = set()
        assert pitch not in pitch_map[instrument_name]
        pitch_map[instrument_name].add(pitch)
    test_set = set(pitch_map.keys())

    reverse_solve = {}
    for x in LEGACY_INSTRUMENTS:
        reverse_solve[x.instrumentName] = x
    # assert test_set == set(reverse_solve.keys())

    with open('intruments_and_ranges.py', 'w') as f:
        print('from typing import List, Tuple', file=f)
        print('', file=f)
        print('from music21 import instrument', file=f)
        print('from music21.instrument import Instrument', file=f)
        print('', file=f)
        print('intruments_ranges: List[Tuple[Instrument, range]] = [', file=f)
        for instrument_name, pitches in pitch_map.items():
            if instrument_name not in test_set:
                continue
            p = list(pitches)
            p.sort()
            p_start = p[0]
            p_stop = p[-1] + 1
            assert len(p) == p_stop - p_start
            class_name = reverse_solve[
                instrument_name
            ].__class__.__name__
            print(
                '    instrument.', class_name, '(), range(', 
                p_start, ', ', p_stop, '), ', 
                sep='', file=f, 
            )
        print(']', file=f)
    print('ok')

main()
