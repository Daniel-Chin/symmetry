from music21 import converter, instrument

TARGET_ROOT = "raising_falling_seq_midi_test_instru/"
MIDI_PATH = "raising_falling_seq_midi/56.mid"
INSTRUMENT_LIST = [
    instrument.Piano(),
    instrument.Harpsichord(),
    instrument.Clavichord(),
    instrument.Celesta(),
    instrument.ElectricPiano(),
    instrument.Harp(),
    instrument.Guitar(),
    instrument.AcousticGuitar(),
    instrument.ElectricGuitar(),
    instrument.FretlessBass(), # 85
    instrument.Banjo(),
    instrument.Lute(),
    instrument.Sitar(),
    instrument.Shamisen(),
    instrument.Koto(),
    instrument.Percussion(),
    instrument.Vibraphone(),
    instrument.Marimba(),
    instrument.Glockenspiel(),
    instrument.ChurchBells(),
    instrument.Gong(),
    instrument.Dulcimer(),
]


ALL_INSTRUMENT_LIST = [
    instrument.Piano(),
    instrument.Harpsichord(),
    instrument.Clavichord(),
    instrument.Celesta(),
    instrument.Sampler(),
    instrument.ElectricPiano(),
    instrument.Organ(),
    instrument.PipeOrgan(),
    instrument.ElectricOrgan(),
    instrument.ReedOrgan(),
    instrument.Accordion(),
    instrument.Harmonica(),
    instrument.Violin(),   # 96
    instrument.Viola(),
    instrument.Violoncello(),
    instrument.Contrabass(),
    instrument.Harp(),
    instrument.Guitar(),
    instrument.ElectricGuitar(),
    instrument.AcousticBass(),
    instrument.ElectricBass(), # 84
    instrument.FretlessBass(), # 85
    instrument.Mandolin(),
    instrument.Ukulele(),
    instrument.Banjo(),
    instrument.Lute(),
    instrument.Sitar(),
    instrument.Shamisen(),
    instrument.Koto(),
    instrument.Flute(),
    instrument.Piccolo(),
    instrument.Recorder(),
    instrument.PanFlute(),
    instrument.Shakuhachi(),
    instrument.Whistle(),
    instrument.Ocarina(),
    instrument.Oboe(),
    instrument.EnglishHorn(),  # 86
    instrument.Clarinet(),
    instrument.BassClarinet(),
    instrument.Bassoon(), # 85
    instrument.Contrabassoon(),
    instrument.Saxophone(), # 85
    instrument.SopranoSaxophone(),
    instrument.AltoSaxophone(),
    instrument.BaritoneSaxophone(), # 73
    instrument.Bagpipes(),
    instrument.Shehnai(),
    instrument.Horn(),
    instrument.Trumpet(),
    instrument.Trombone(),
    instrument.BassTrombone(),
    instrument.Tuba(), #73
    instrument.Vibraphone(),
    instrument.Marimba(),
    instrument.Xylophone(),
    instrument.Glockenspiel(),
    instrument.ChurchBells(),
    instrument.TubularBells(),
    instrument.Gong(),
    instrument.Handbells(),
    instrument.Dulcimer(),
    instrument.SteelDrum(),
    instrument.Timpani(),
    instrument.Kalimba(),
    instrument.Vibraslap(),
    instrument.Cowbell(),
    instrument.Soprano(),
    instrument.Bass()
]


def replace_instrument(source_midi_path, target_instrument, target_path):
    s = converter.parse(source_midi_path)
    for el in s.recurse():
        if 'Instrument' in el.classes:  # or 'Piano'
            el.activeSite.replace(el, target_instrument)
    s.write('midi', target_path)


if __name__ == "__main__":
    i = 0
    for ins in ALL_INSTRUMENT_LIST:
        replace_instrument(MIDI_PATH, ins, f'{TARGET_ROOT}{i}-{ins.instrumentName}.mid')
        i += 1