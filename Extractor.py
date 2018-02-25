'''
Source: https://github.com/PortfolioCollection/Character_Recogniser
Authors: Muratovm, cheesywow.
Adapted by: luid101
'''

import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as numpy

def getImage(filename):
    """Returns a two dimensional array of a chosen image's pixels"""
    try:
        image = Image.open(filename, 'r')
    except Exception as e:
        print(e)
    
    return image


def selectImage():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    file = open(file_path)
    filename = file.name
    print(file.name)
    
    try:
        image = Image.open(filename, 'r')
    except Exception as e:
        print(e)
    
    return image

def ImageToArray(image):
    width, height = image.size
    pixel_values = list(image.getdata())
    color_array = []
    for h in range(height):
        color_array.append([])
        for w in range(width):
            color_array[h].append(pixel_values[h*width+w])
    return color_array


def ImageToMatrix(image):
    array = numpy.asarray(image)
    return array

if __name__ == "__main__":
    image = getImage()
    #matrix = ImageToMatrix(image)
    #print(matrix)
    array = ImageToArray(image)
    for row in array:
        print(row)
    exit = input("Type any enter to exit")
