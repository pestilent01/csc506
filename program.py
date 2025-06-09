'''This module is the main program that uses Huffman coding to compress text and images.'''
import base64
import io
from huffmancoding import huffmanCoding, encode, decode
from metrics import measureTime, measureSpace, getCompressionRatio
from training import getHuffmanCodeTextModel, getHuffmanCodeImageModel, getImageTrainingData, trainHuffmanCodeImage, trainHuffmanCodeText, convertToBase64, convertToImage, trainHuffmanCodeImageHash
import os
import numpy as np
from PIL import Image
from charset_normalizer import detect
if __name__ == "__main__":
    '''This function will ask the user for what they want to do. 1 - Train the Huffman coding model, 2 - Train the Huffman coding model with images, 3 - Compress text, 4 - Compress an image'''
    print("Welcome to the Huffman coding program!")
    print("1 - Train the Huffman coding model for text compression")
    print("2 - Train the Huffman coding model for image compression")
    print("3 - Compress some text")
    print("4 - Compress an image")
    print("Q - Quit")
    
    choice = input("Enter your choice: ")
    while choice != 'Q':
        if choice == '1':
            print("Training the Huffman coding model for text compression...")
            trainHuffmanCodeText()
            print("Training complete!")
        elif choice == '2':
            print("Training the Huffman coding model for image compression...")
            #trainHuffmanCodeImageHash()
            trainHuffmanCodeImage()
            print("Training complete!")
        elif choice == '3':
            print("Compressing some text...")
            text_to_compress = input("Enter the text to compress: ")

            huffman_code_text = getHuffmanCodeTextModel()
            encoded_text = encode(text_to_compress, huffman_code_text)
            print("Encoded text: ", encoded_text[:100])
            print("Original text length(bits): ", len(text_to_compress)*8)
            print("Encoded text length(bits): ", len(encoded_text))
            print("Compression ratio: ", getCompressionRatio(len(text_to_compress)*8, len(encoded_text)), "%")
        elif choice == '4':
            '''User enter a path of an image to compress'''
            image_path = input("Enter the path of the image to compress: ")
            if os.path.exists(image_path):
                # Open the image file
                image = Image.open(image_path)
                dimensions = image.size
                base64Image = convertToBase64(image)
                huffman_code_image = getHuffmanCodeImageModel()
                encoded_image = encode(base64Image, huffman_code_image)
                print ("Original image size(bits): ", os.path.getsize(image_path)*8)
                print("Encoded image: ", encoded_image[:100])
                print("Original image length(bits): ", len(base64Image)*8)
                print("Encoded image length(bits): ", len(encoded_image))
                print("Compression ratio: ", getCompressionRatio(len(base64Image)*8, len(encoded_image)), "%")
                decoded_image = decode(encoded_image, huffman_code_image)
                print("Decoded image: ", decoded_image[:100])
                # Convert base64 encoded decoded_image to Image object
                numpy_data = base64.b64decode(decoded_image)
                image_data = np.frombuffer(numpy_data, dtype=np.uint8)
                image_data = image_data.reshape(dimensions[1], dimensions[0], 3)
                decoded_image = Image.fromarray(image_data)
                decoded_image.save("decoded_image.png", format="PNG")
                print("Decoded image saved as decoded_image.png")



            else:
                print("File not found!")
        elif choice == 'Q':
            print("Goodbye!")
            break
        choice = input("Enter your choice: ")



