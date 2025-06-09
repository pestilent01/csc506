import io
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from huffmancoding import huffmanCoding, encode, decode
import base64
from PIL import Image
import random
def getTextTrainingData(numberOfReviews=1000):

    ds,info = tfds.load('imdb_reviews', split='train', as_supervised=True, with_info=True)
    text = ''
    data = ds.take(numberOfReviews)
    validationData = [text.numpy().decode("utf-8") for text, label in data]

    for review in ds.take(numberOfReviews):
        text += review[0].numpy().decode('utf-8')
    return text+" ", validationData

def getImageTestData():
    ds,info = tfds.load('cifar10', split='train', as_supervised=True, with_info=True)
    image = ds.take(1)
    image = next(iter(image))[0]
    base64Image = convertToBase64(image)
    image = tf.image.rgb_to_grayscale(image)
    tmp = image.numpy()
    dimensions = tmp.shape
        
    return base64Image, dimensions

def getRandomImageGeneratorTrainingData():
    images = []
    for i in range(1000):
        # Generate a random width and height between 16 and 64
        width = random.randint(16, 64)
        height = random.randint(16, 64)

        # Generate a random image with 3 channels (RGB)
        image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        # Convert to string list for compatibility with other functions
        stringList = convertToString(image.flatten())
        size = image.size

        images.append((stringList, size))

    data =[]
    for image in images:
        data.append(image[0])
    return data, images
    


def getImageTrainingData():
    '''Get image training data from tensorflow cifar10 dataset. Combine all images into a single string.'''
    ds,_ = tfds.load('cifar10', split='train', as_supervised=True, with_info=True)
    images = []

    for image, label in ds.take(1000):
        #Convert to Image object
        image = image.numpy()
        size = image.shape[0]*image.shape[1]*image.shape[2]
        #img = Image.fromarray(image)
        stringList = convertToString(image.flatten())
        images.append((stringList, size))
    
    #get the stringList from images
    data = []
    for image in images:
        data.append(image[0])

    return data, images

def convertToString(arrayData):
    '''Convert a vector of integers to a string.'''
    data = [ str(i) for i in arrayData]
    return data

def convertToImage(stringData):
    '''Convert string of comma delimited integer values into an integer array, then reshape to (32,32,3).'''
    data = stringData.split(',')
    data = [int(i) for i in data]
    #convert the list of integers to numpy array
    image_array = np.array(data, dtype=np.uint8)

    image_array = image_array.reshape((32, 32, 3))
    # Convert the numpy array to a PIL Image
    image = Image.fromarray(image_array)
    return image


def convertToBase64(image):
    result = base64.b64encode(image.tobytes()).decode('utf-8')

    return result

def convertToPixel(image):
    #every three values on the image list is a pixel value, combine into one single string delimted by "-"
    result =[]
    for i in range(0, len(image), 3):
        pixel = str(image[i]) + "-" + str(image[i+1]) + "-" + str(image[i+2])
        result.append(pixel)
    return result

def restorePixelToList(image):
    #restore a list of string that are delimited by "-", parse into three integers
    result = []
    pixels = image.split(' ')
    for pixel in pixels:
        pixel_channel = pixel.split('-')
        result.extend(pixel_channel)
    return result

def convertToVector(stringData):
    '''Convert a comma separated string to a vector of integers.'''
    result = list(map(int, stringData.split(',')))
    return result

def getAudioTrainingData():
    '''Get audio training data from tensorflow speech_commands dataset. Combine all audio files into a single string.'''
    ds,info = tfds.load('speech_commands', split='train', as_supervised=True, with_info=True)
    audio = []
    for audio_file, label in ds.take(1000):
        audio.append(audio_file.numpy())
    
    return audio

def trainHuffmanCodeText():
    '''Train the Huffman code on text data and save it to a file.'''
    training_text, _ = getTextTrainingData()
    huffman_code_text = huffmanCoding(training_text)
    with open('./data/huffman_code_text.data', 'w',encoding="utf-8") as file:
        for char, code in huffman_code_text.items():
            file.write(f"{char}: {code}\n")

def getHuffmanCodeTextModel():
    '''Get the Huffman code for text data.'''
    huffman_code = {}
    with open('./data/huffman_code_text.data', 'r', encoding="utf-8") as file:
        for line in file:
            char, code = line.strip().split(': ')
            huffman_code[char] = code
    return huffman_code
def trainHuffmanCodeImage():
    '''Train the Huffman code on image data and save it to a file.'''
    training_image, _ = getRandomImageGeneratorTrainingData()
    flattenData = []
    for image in training_image:
        print("Training image size: ", len(image))
        pixel = convertToPixel(image)
        flattenData.extend(pixel)
        
        break
    
    huffman_code_image = huffmanCoding(flattenData)
    with open('./data/huffman_code_image.data', 'w') as file:
        for char, code in huffman_code_image.items():
            file.write(f"{char}: {code}\n")
def trainHuffmanCodeImageHash():
    '''Train the Huffman code on image data and save it to a file.'''
    training_image, _ = getRandomImageGeneratorTrainingData()
    flattenData = []
    for image in training_image:
        pixel = convertToPixel(image)
        flattenData.extend(pixel)
        break
    
    huffman_code_image = huffmanCoding(flattenData)
    huffman_code_image_hash = [None] * len(huffman_code_image)
    #create a hash
    hash_index = 0
    for char, code in huffman_code_image.items():
        #hash the character to a number
        hash_index = hash(char) % len(huffman_code_image_hash)
        #convert string of "0" and "1" to bytes
        code_bytes = bytes(int(code[i:i+8], 2) for i in range(0, len(code), 8))
        #convert bytes to string

        huffman_code_image_hash[hash_index] = code_bytes
        
    
    with open('./data/huffman_code_image_hash.data', 'w') as file:
        for code in huffman_code_image_hash:
            file.write(f"{code}\n")

def getHuffmanCodeImageModel():
    '''Get the Huffman code for image data.'''
    huffman_code = {}
    with open('./data/huffman_code_image.data', 'r') as file:
        for line in file:
            char, code = line.strip().split(': ')
            huffman_code[char] = code
    return huffman_code