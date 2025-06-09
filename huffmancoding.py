import heapq


def getFrequency(text):
    """Get the frequency of each character in the text."""
    frequency = {}
    for char in text:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
    return frequency


def buildHuffmanTree(frequency):
    """Build the Huffman tree from the frequency dictionary."""
    heap = [[weight, [char, ""]] for char, weight in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
     #   print(f'Lowest: {lo}')
        hi = heapq.heappop(heap)

        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    item = heapq.heappop(heap)
    return sorted(item[1:], key=lambda p: (len(p[-1]), p))

def huffmanCoding(text):
    """Perform Huffman coding on the given text."""
    frequency = getFrequency(text)
    huffman_tree = buildHuffmanTree(frequency)
    huffman_code = {char: code for char, code in huffman_tree}
    return huffman_code

def encode(text, huffman_code):
    """Encode the text using the Huffman tree."""
    encoded_text = ''.join(huffman_code[char] for char in text)
    return encoded_text

def decode(encoded_text, huffman_code, separator=''):
    """Decode the encoded text using the Huffman tree."""
    reverse_huffman_code = {code: char for char, code in huffman_code.items()}
    current_code = ""
    decoded_text = ""
    for bit in encoded_text:
        current_code += bit
        if current_code in reverse_huffman_code:
            decoded_text += (separator + reverse_huffman_code[current_code]) if len(decoded_text) > 0 else reverse_huffman_code[current_code]
            current_code = ""
    return decoded_text

def writeHuffmanCodeToFile(huffman_code, filename):
    """Write the Huffman code to a file."""
    with open(filename, 'w', encoding="utf-8") as file:
        for char, code in huffman_code.items():
            file.write(f"{char}: {code}\n")

def readHuffmanCodeFromFile(filename):
    """Read the Huffman code from a file."""
    huffman_code = {}
    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            char, code = line.strip().split(': ')
            huffman_code[char] = code
    return huffman_code

if __name__ == "__main__":
    # Example usage, test the Huffman coding functions
    text = "hello world"
    huffman_code = huffmanCoding(text)
    print("Huffman Code: ", huffman_code)
    