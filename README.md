# pixels-n-pitches
Ever heard a song and wish there was art that matched the vibe of that song? Well fret not (pun intended), now you can.

## Creators
Lyndsey Becker, Victoria Cabral, Nyx Harris-Palmer

## What was this Created For?
This was created for Technica 2022

## What does this code do?
This uses AI to take midi files of songs and create art using them

## How to use the code
Requires Python 3.10+, along with the `Tensorflow`, `NumPy`, `Pillow`, and `Mido` packages.
```
usage: pixels-n-pitches.py [-h] --midifiles MIDIFILES [MIDIFILES ...]

Turn MIDI files of your favorite songs into AI art!

options:
  -h, --help            show this help message and exit
  --midifiles MIDIFILES [MIDIFILES ...]
                        MIDI file to process
```
Example (on a Linux machine): `python pixels-n-pitches.py --midifiles Midi\ Files/Moondance.mid`