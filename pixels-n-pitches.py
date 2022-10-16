import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path

import numpy as np
from tensorflow import keras
from PIL import Image

from midi2np import midi_to_np_arr


TRAINING_DATA_DICT = {
    'A#Minor.mid': 'A#Minor.jpg',
    'AbMajor.mid': 'AbMajor.jpg',
    'AHarmonicMinor.mid': 'AHarmonicMinor.jpg',
    'AllStar.mid': 'AllStar.jpg',
    'AMajor.mid': 'AMajor.jpg',
    'AMinor.mid': 'AMinor.jpg',
    'AnotherOneBitesTheDust.mid': 'AnotherOneBitesTheDust.jpg',
    'BbMajor.mid': 'BbMajor.jpg',
    'BHarmonicMinor.mid': 'BHarmonicMinor.jpg',
    'BMajor.mid': 'BMajor.jpg',
    'Bminor.mid': 'Bminor.jpg',
    'C#HarmonicMinor.mid': 'C#HarmonicMinor.jpg',
    'CallMeMaybe.mid': 'CallMeMaybe.jpg',
    'CantTouchThis.mid': 'CantTouchThis.jpg',
    'CMajor.mid': 'CMajor.jpg',
    'CMinor.mid': 'CMinor.jpg',
    'DbMajor.mid': 'DbMajor.jpg',
    'Despacito.mid': 'Despacito.jpg',
    'DHarmonicMinor.mid': 'DHarmonicMinor.jpg',
    'DMajor.mid': 'DMajor.jpg',
    'DMinor.mid': 'DMinor.jpg',
    'EbMajor.mid': 'EbMajor.jpg',
    'EHarmonicMinor.mid': 'EHarmonicMinor.jpg',
    'EMajor.mid': 'EMajor.jpg',
    'EMinor.mid': 'EMinor.jpg',
    'F#HarmonicMinor.mid': 'F#HarmonicMinor.jpg',
    'FHarmonicMinor.mid': 'FHarmonicMinor.jpg',
    'FMajor.mid': 'FMajor.jpg',
    'FMinor.mid': 'FMinor.jpg',
    'Fur Elise.mid': 'FurEliseImage.jpg',
    'GbMajor.mid': 'GbMajor.jpg',
    'GHarmonicMinor.mid': 'GHarmonicMinor.jpg',
    'GMajor.mid': 'GMajor.jpg',
    'Gminor.mid': 'Gminor.jpg',
    'Imagine dragons - Radioactive.mid': 'BeliverImage.jpg',
    'ImYours.mid': 'ImYours.jpg',
    'Margaritaville.mid': 'Margaritaville.jpg',
    'Moondance.mid': 'Moondance.jpg',
    'Never-Gonna-Give-You-Up-3.mid': 'NeverGonnaGiveYouUpImage.jpg',
    'PhantomoftheOpera.mid': 'PhantomoftheOpera.jpg',
    'PianoMan.mid': 'PianoMan.jpg',
    'potter.mid': 'HarryPotterImage.jpg',
    'potter.mid': 'HarryPotterImage.jpg',
    'RunninWithTheDevil.mid': 'RunninWithTheDevil.jpg',
    'SoulMan.mid': 'SoulMan.jpg',
    'StealMyGirl.mid': 'StealMyGirl.jpg',
    'SurfinUSA.mid': 'SurfinUSA.jpg',
    'ThatsLife.mid': 'ThatsLife.jpg',
    'The-Avengers.mid': 'AvengersImage.jpg',
    'TheMiddle.mid': 'TheMiddle.jpg',
    'ThreeLittleBirds.mid': 'ThreeLittleBirds.jpg',
    'titanic-3.mid': 'TitanicImage.jpg',
    'Wonderwall.mid': 'Wonderwall.jpg',
    "Gangsta's-Paradise-1.mid": 'GasntaParadiseImage.jpg',
}


def assemble_model():
    inputs = keras.Input(shape=(1024,))
    dense = keras.layers.Dense(2048,
        activation="tanh")
    x = dense(inputs)
    dense2 = keras.layers.Dense(2560, activation="tanh")
    y = dense2(x)
    dense3 = keras.layers.Dense(3072, activation="sigmoid")
    outputs = dense3(y)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Turn MIDI files of your favorite songs into AI art!'
    )

    def valid_mid_file(path):
        f = Path(path)
        if not f.is_file():
            raise ValueError
        if f.suffix not in ('.mid', '.midi'):
            raise ValueError
        return path

    parser.add_argument(
        '--midifiles',
        type=valid_mid_file,
        required=True,
        nargs='+',
        help='MIDI file to process'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    midifiles = args.midifiles

    train_paths_i = ('Midi Files/'+p for p in TRAINING_DATA_DICT.keys())
    train_paths_o = ('Images/'+p for p in TRAINING_DATA_DICT.values())
    input_train_notes = np.array(list(map(lambda m: midi_to_np_arr(m, 1024, shuffle=True), train_paths_i)))
    output_train_imgs = np.array(list(map(lambda p: np.reshape(np.array(Image.open(p), dtype=float), 3072), train_paths_o)))

    model = assemble_model()
    model.compile(
        optimizer="rmsprop",
        loss="cosine_similarity",
        metrics=['Precision']
    )
    model.fit(input_train_notes, output_train_imgs)
    model.evaluate(input_train_notes, output_train_imgs)

    for midi_file in midifiles:
        predictions = model.predict(midi_to_np_arr(midi_file, 1024).reshape((1, 1024)))
        img_pred = np.reshape(255*predictions, (32, 32, 3))
        img_final = Image.fromarray(img_pred, mode='RGB')
        img_final.show()
        img_final.save('Output Images/'+midi_file.rpartition('/')[-1].replace('.mid', '.jpg'))


if __name__ == '__main__':
    main()