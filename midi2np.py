import mido
import numpy as np
mid = mido.MidiFile('Never-Gonna-Give-You-Up-3.mid', clip=True)
merged = mido.merge_tracks(mid.tracks)
messages = list(m.note for m in filter(lambda msg: msg.type=='note_on', merged))
notes = np.empty(len(messages), dtype=object)
notes[:] = messages
# print(notes.tolist())
# print(len(notes))