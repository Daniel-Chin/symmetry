from typing import List

from music21 import instrument
from music21.instrument import Instrument

LEGACY_INSTRUMENTS: List[Instrument] = [
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
