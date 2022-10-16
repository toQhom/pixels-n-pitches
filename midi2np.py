import mido
import numpy as np
import random

def midi_to_np_arr(path_to_mid, length, shuffle=False):
    mid = mido.MidiFile(path_to_mid, clip=True)
    messages = []
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on':
                messages.append(msg.note)
                if len(messages) == length:
                    break
        if len(messages) == length:
            break
    while len(messages) < length:
        messages.append(-1)
    if shuffle:
        random.shuffle(messages)
    notes = np.fromiter(messages, dtype=float)
    return notes