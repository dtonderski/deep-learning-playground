from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.applications.vgg16 import VGG16
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from os import listdir
from pickle import load, dump

from numpy import array

def load_file(filepath = "data/text/Flickr8k.lemma.token.txt"):
    f = open(filepath, 'r')
    text = f.read()
    f.close()
    return text

def clean_text(text):
    cleaned_text = ''
    for line in text.split('\n'):
        if len(line) == 0:
            continue
        cleaned_text += "startseq "
        desc = line.split('\t')[1]
        for word in desc.split(' '):
            word = word.translate(str.maketrans('', '', punctuation))
            if len(word) > 1:
                cleaned_text += word.lower() + ' '
        cleaned_text += "endseq\n"
    return cleaned_text

def create_tokenizer(cleaned_text):
    text = ' '.join(cleaned_text.split('\n'))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text.split(' '))
    return tokenizer

# Create a dictionary with keys photoid and values features from the photos in the directory
def extract_features(directory):
    features = dict()
    vggmodel = VGG16()
    model = Model(inputs=vggmodel.inputs, outputs=vggmodel.layers[-2].output)
    for filename in listdir(directory):
        image = load_img(directory + '/' + filename, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = image_array.reshape(1, 224, 224, 3)
        feature = model.predict(image_array)
        name = filename.split('.')[0]
        features[name] = feature
        print(filename)
    return features

def extract_feature(path):
    vggmodel = VGG16()
    model = Model(inputs=vggmodel.inputs, outputs=vggmodel.layers[-2].output)
    image = load_img(path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = image_array.reshape(1, 224, 224, 3)
    feature = model.predict(image_array)
    return feature

def load_photo_names_for_set(set_name = "train"):
    f = open("data/text/Flickr_8k.%sImages.txt" % set_name, 'r')
    text = f.read()
    photo_names = set()
    for line in text.split("\n"):
        if len(line) == 0:
            continue
        photo_name = line.split(".")[0]
        photo_names.add(photo_name)
    return photo_names


def create_descriptions_dictionary_for_set(photo_names, filepath = "data/text/Flickr8k.lemma.token.txt"):
    descriptions = dict()
    text = load_file(filepath)
    for line in text.split('\n'):
        if len(line) == 0:
            continue
        photo_id, desc = line.split("\t")
        photo_name = photo_id.split(".")[0]
        if not photo_name in photo_names:
            continue
        cleaned_desc = ""
        for word in desc.split(' '):
            word = word.translate(str.maketrans('', '', punctuation))
            if len(word) > 1:
                cleaned_desc += word.lower() + ' '
        descriptions[photo_name] = "startseq " + cleaned_desc + "endseq"
    return descriptions


def create_data_for_network(descriptions, max_length = 34, vocab_size = 6734):
    features = load_features()
    tokenizer = load_tokenizer()
    x1, x2, y = list(), list(), list()
    for key,description in descriptions.items():
        feature = features[key][0]
        sequence = tokenizer.texts_to_sequences([description])[0]
        for i in range(len(sequence)):
            current_sequence, next_value = sequence[0:i], sequence[i]
            current_sequence = pad_sequences([current_sequence], max_length)[0]
            next_value = to_categorical([next_value], num_classes=vocab_size)[0]
            x1.append(feature)
            x2.append(current_sequence)
            y.append(next_value)
    return array(x1), array(x2), array(y)

def save_features(features, filepath = 'data/photos/features.pkl'):
    file = open(filepath, 'wb')
    dump(features, file)

def load_features(filepath = 'data/photos/features.pkl'):
    file = open(filepath, 'rb')
    return load(file)

def save_tokenizer(tokenizer, filepath = 'data/text/tokenizer.pkl'):
    file = open(filepath, 'wb')
    dump(tokenizer, file)

def load_tokenizer(filepath = 'data/text/tokenizer.pkl'):
    file = open(filepath, 'rb')
    return load(file)