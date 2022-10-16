import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
from midi2np import midi_to_np_arr
"""
Notes:

Input: 1024 x 1 array of MIDI note values

Hidden: 2048 x 1 array of who know what

Output: 3072 x 1 (32x32x3 reshape) array of pixels for image

Activation Functions:

"""

inputs = keras.Input(shape=(1024,))
dense = layers.Dense(2048, activation="relu")
x = dense(inputs)

dense2 = layers.Dense(3072, activation="sigmoid")
outputs = dense2(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

midi_to_images = {
    'potter.mid': 'HarryPotterImage.jpg',
#     'SmellsLikeTeenSpirit.mid': '',
#     'ItsASin.mid': '',
    'Fur Elise.mid': 'FurEliseImage.jpg',
    'titanic-3.mid': 'TitanicImage.jpg',
    'Imagine dragons - Radioactive.mid': 'BeliverImage.jpg',
    'Never-Gonna-Give-You-Up-3.mid': 'NeverGonnaGiveYouUpImage.jpg',
#     'Rasputin.mid': '',
    "Gangsta's-Paradise-1.mid": 'GasntaParadiseImage.jpg',
    'The-Avengers.mid': 'AvengersImage.jpg',
#     'BeforeHeCheats.mid': '',
#     'StayinAlive(2).mid': '',
    'A#Minor.mid': 'A#Minor.jpg',
    'AbMajor.mid': 'AbMajor.jpg',
    'AHarmonicMinor.mid': 'AHarmonicMinor.jpg',
    'AMajor.mid': 'AMajor.jpg',
    'AMinor.mid': 'AMinor.jpg',
    'BbMajor.mid': 'BbMajor.jpg',
    'BHarmonicMinor.mid': 'BHarmonicMinor.jpg',
    'BMajor.mid': 'BMajor.jpg',
    'Bminor.mid': 'Bminor.jpg',
    'C#HarmonicMinor.mid': 'C#HarmonicMinor.jpg',
    'CMajor.mid': 'CMajor.jpg',
    'CMinor.mid': 'CMinor.jpg',
    'DbMajor.mid': 'DbMajor.jpg',
    'DHarmonicMinor.mid': 'DHarmonicMinor.jpg',
    'DMajor.mid': 'DMajor.jpg',
    'DMinor.mid': 'DMinor.jpg',
    'EHarmonicMinor.mid': 'EHarmonicMinor.jpg',
    'EMajor.mid': 'EMajor.jpg',
    'EMinor.mid': 'EMinor.jpg',
    'EbMajor.mid': 'EbMajor.jpg',
    'F#HarmonicMinor.mid': 'F#HarmonicMinor.jpg',
    'FHarmonicMinor.mid': 'FHarmonicMinor.jpg',
    'FMajor.mid': 'FMajor.jpg',
    'FMinor.mid': 'FMinor.jpg',
    'GHarmonicMinor.mid': 'GHarmonicMinor.jpg',
    'GMajor.mid': 'GMajor.jpg',
    'GbMajor.mid': 'GbMajor.jpg',
    'Gminor.mid': 'Gminor.jpg',
    'potter.mid': 'HarryPotterImage.jpg',
}

train_paths_i = ('Midi Files/'+p for p in midi_to_images.keys())
train_paths_o = ('Images/'+p for p in midi_to_images.values())

input_train_notes = np.array(list(map(lambda m: midi_to_np_arr(m, 1024), train_paths_i)))
output_train_imgs = np.array(list(map(lambda p: np.reshape(np.array(Image.open(p)), 3072), train_paths_o)))
# print(input_train_notes.shape)
# print(output_train_imgs.shape)


model.compile(loss="mean_squared_error")
model.fit(input_train_notes, output_train_imgs)

print("Evaluate on test data")
results = model.evaluate(input_train_notes, output_train_imgs)
print("test loss, test acc:", results)

print("Generate predictions for 3 samples")
predictions = model.predict(midi_to_np_arr('New Training Files/AllStar.mid', 1024).reshape((1, 1024)))
print("predictions shape:", predictions.shape)

img_pred = np.reshape(predictions, (32, 32, 3))
print(img_pred)
img_final = Image.fromarray(img_pred, mode='RGB')
img_final.show()