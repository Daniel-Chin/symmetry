from typing import List, Tuple

from music21 import instrument
from music21.instrument import Instrument

intruments_ranges: List[Tuple[Instrument, range]] = [
    instrument.Accordion(), range(58, 97), 
    instrument.AcousticBass(), range(48, 97), 
    instrument.Banjo(), range(36, 97), 
    instrument.BaritoneSaxophone(), range(36, 73), 
    instrument.Bassoon(), range(36, 85), 
    instrument.Celesta(), range(36, 97), 
    instrument.ChurchBells(), range(36, 97), 
    instrument.Clarinet(), range(41, 85), 
    instrument.Clavichord(), range(36, 85), 
    instrument.Dulcimer(), range(36, 85), 
    instrument.ElectricBass(), range(40, 85), 
    instrument.ElectricGuitar(), range(36, 97), 
    instrument.ElectricOrgan(), range(36, 97), 
    instrument.ElectricPiano(), range(36, 97), 
    instrument.EnglishHorn(), range(36, 86), 
    instrument.Flute(), range(48, 97), 
    instrument.FretlessBass(), range(36, 85), 
    instrument.Glockenspiel(), range(36, 97), 
    instrument.Guitar(), range(36, 97), 
    instrument.Handbells(), range(36, 85), 
    instrument.Harmonica(), range(36, 97), 
    instrument.Harp(), range(36, 97), 
    instrument.Harpsichord(), range(36, 97), 
    instrument.Horn(), range(36, 97), 
    instrument.Kalimba(), range(36, 97), 
    instrument.Koto(), range(36, 97), 
    instrument.Lute(), range(36, 97), 
    instrument.Mandolin(), range(36, 97), 
    instrument.Marimba(), range(36, 97), 
    instrument.Oboe(), range(36, 97), 
    instrument.Ocarina(), range(36, 97), 
    instrument.Organ(), range(36, 97), 
    instrument.PanFlute(), range(36, 97), 
    instrument.Piano(), range(36, 97), 
    instrument.Piccolo(), range(48, 97), 
    instrument.Recorder(), range(36, 97), 
    instrument.ReedOrgan(), range(36, 97), 
    instrument.Sampler(), range(36, 97), 
    instrument.Saxophone(), range(36, 85), 
    instrument.Shakuhachi(), range(36, 97), 
    instrument.Shamisen(), range(36, 97), 
    instrument.Shehnai(), range(36, 97), 
    instrument.Sitar(), range(36, 97), 
    instrument.SopranoSaxophone(), range(36, 97), 
    instrument.SteelDrum(), range(36, 97), 
    instrument.Timpani(), range(36, 97), 
    instrument.Trombone(), range(36, 97), 
    instrument.Trumpet(), range(36, 97), 
    instrument.Tuba(), range(36, 73), 
    instrument.Vibraphone(), range(36, 97), 
    instrument.Viola(), range(36, 97), 
    instrument.Violin(), range(36, 97), 
    instrument.Violoncello(), range(36, 97), 
    instrument.Whistle(), range(48, 97), 
    instrument.Xylophone(), range(36, 97), 
]
