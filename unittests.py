import unittest
from huffmancoding import huffmanCoding, encode, decode, getFrequency
from training import getTextTrainingData, getImageTrainingData, convertToPixel, convertToString, restorePixelToList, getRandomImageGeneratorTrainingData
from metrics import getCompressionRatio, measureTime, measureSpace
import tensorflow as tf
import os


class TestHuffmanCoding(unittest.TestCase):

    def setUp(self):
        # This method will be run before each test
        trainingText, validationText = getTextTrainingData()
        self.training_text = trainingText
        self.validation_text = validationText

        print("Training text size: ", len(self.training_text))
        self.huffman_code_text = huffmanCoding(self.training_text)

        self.training_image, self.validation_image = getRandomImageGeneratorTrainingData()

        flattenData = []
        for image in self.training_image:
            pixel = convertToPixel(image)
            flattenData.extend(pixel)

        #self.training_image = flattenData
        self.flattenData = flattenData
        print("Training image size: ", len(self.training_image))
        self.huffman_code_image = huffmanCoding(flattenData)
  
        #write header to file
        self.writeHeader()

    def convertCorpusToWordList(self, corpus):
        '''Convert a corpus of text to a list of words.'''
        # Convert the corpus to a list of words
        word_list = corpus.split(" ")
        result = []
        #add a space element between each word in the list except the last word
        for i in range(len(word_list)):
            result.append(word_list[i])
            if i < (len(word_list)-1):
                result.append(' ')

        return result
    
    def test_huffman_code_image(self):
        '''Test various size number of unique pixel in the image to build out the huffmancode tree, then measure the time and space complexity of building out the huffman code tree.'''
        frequency = getFrequency(self.flattenData)
        for i in range(1, len(frequency), 10000):
            # Get i items from the frequency dictionary
            subset_frequency = dict(list(frequency.items())[:i])
            #print("Subset frequency: ", subset_frequency)
            input_size = len(subset_frequency)
            # Build Huffman code for the subset
            huffman_code_subset = huffmanCoding(subset_frequency)
            # Measure time and space complexity of encoding and decoding the image
            tree, duration = measureTime(huffmanCoding, subset_frequency)
            _, current, peak = measureSpace(huffmanCoding, subset_frequency)
            
            # Log results
            self.writeLog("test_huffman_code_tree_images", input_size, len(tree), duration, current, peak)

    def test_huffman_code_text(self):
        '''Test various size number of unique characters in the text to build out the huffmancode tree, then measure the time and space complexity of building out the huffman code tree.'''
        frequency = getFrequency(self.training_text)

        for i in range(10, len(frequency), 10):
            # Get i items from the frequency dictionary
            subset_frequency = dict(list(frequency.items())[:i])

            input_size = len(subset_frequency)

            # Measure time and space complexity of encoding and decoding the text
            tree, duration = measureTime(huffmanCoding, subset_frequency)
            _, current, peak = measureSpace(huffmanCoding, subset_frequency)
            
            # Log results
            self.writeLog("test_huffman_code_tree", input_size, len(tree), duration, current, peak)


    def test_text_compression(self):

        for test_text in self.validation_text:
            # Measure input size
            input_size = len(test_text)*8 # Convert bytes to bits

            word_list = test_text
            encoded_text = encode(word_list, self.huffman_code_text)
            encoded_text, duration = measureTime(encode, word_list, self.huffman_code_text)
            _, current, peak = measureSpace(encode, word_list, self.huffman_code_text)
            # Measure output size
            output_size = len(encoded_text)
            
            # Decode the compressed text
            decoded_text = decode(encoded_text, self.huffman_code_text)

            # Check if the decoded text matches the original text
            self.assertEqual(test_text, decoded_text)

            self.writeLog("test_text_compression",input_size, output_size, duration, current, peak) 
           

    def test_image_compression(self): 

        for test_image in self.training_image:
            # Measure input size
            input_size = len(test_image)*8 # Convert bytes to bits

            test_data = convertToPixel(test_image)

            encoded_image, duration = measureTime(encode, test_data, self.huffman_code_image)
            # Measure output size
            output_size = len(encoded_image)
            _, current, peak = measureSpace(encode, test_data,  self.huffman_code_image)
            # Decode the compressed image
            decoded_image = decode(encoded_image,  self.huffman_code_image,' ')
            #expend it back to list of integers
            decoded_image_restored = restorePixelToList(decoded_image)

            self.assertEqual(test_image, decoded_image_restored)
          
            #self.printOutput("test_image_compression", input_size, output_size, duration, current, peak, test_image, encoded_image, decoded_image_restored)
            self.writeLog("test_image_compression",input_size, output_size, duration, current, peak)




    def writeHeader(self):
        if not os.path.exists('results.csv'):
            # Create the file and write the header
            with open('results.csv', 'w') as result_file:
                result_file.write("Test Name,Input Size,Output Size,Compression Ratio,Time Taken,Current Memory,Peak Memory\n")
       
    def writeLog(self,test_name, input_size, output_size, duration, current, peak):
        with open('results.csv', 'a') as result_file:
            result_file.write(f"{test_name},{input_size},{output_size},{getCompressionRatio(input_size, output_size):2f},{duration},{current},{peak}\n")
    
    def printOutput(self, test_name, input_size, output_size, duration, current, peak, test_text, encoded_text, decoded_text):
        print(f"Test Name: {test_name}")
        print(f"Current memory usage: {current} bytes")
        print(f"Peak memory usage: {peak} bytes")
        print(f"Time taken: {duration} seconds")
        print(f"Input size: {input_size} bits")
        print(f"Output size: {output_size} bits")
        print(f"Compression ratio: {getCompressionRatio(input_size, output_size):.2f}")
        print(f"Original text: {test_text[:10]}...")
        print(f"Encoded text: {encoded_text[:10]}...")
        print(f"Decoded text: {decoded_text[:10]}...")



if __name__ == '__main__':
    unittest.main()
"""    training_image, validation_image = getImageTrainingData()

    flattenData = []
    for image in training_image:
        pixel = convertToPixel(image)
        flattenData.extend(pixel)

    print(len(flattenData))
    print(len(getFrequency(flattenData))) """
