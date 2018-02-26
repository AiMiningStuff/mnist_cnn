'''
Source: https://github.com/PortfolioCollection/Character_Recogniser
Authors: Muratovm, cheesywow.
Adapted by: luid101
'''

#-------CNN IMPORST-----#
from keras.models import model_from_json

#----USEFULL CLASSES---#
import Extractor

'''
#--------Hopping-------#
import Hop
import os
import sys

#----CUSTOM CLASSES-----#
os.chdir("..")
Hop.set_project_path()
sys.path.append(os.getcwd()+"/-Averaged Approach-")
sys.path.append(os.getcwd()+"/-KNN Approach-")
import Averaged_Tester
import KNN_Tester
Hop.go_to_core()
'''

#---SUPPORT LIBRARIES---#
from tkinter import *
from PIL import Image
from PIL import ImageDraw
import numpy as np


array = np.full((280,280),255)
img = Image.fromarray(array.astype(np.uint8))
class Paint():

    def __init__(self):
        
        # set brush size
        self.brush_size = 30

        # Load Model from disk
        # load json and create model
        json_file = open('model_struct.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # load weights into new model
        loaded_model.load_weights("model_weights.h5")
        
        self.model = loaded_model
        print("Loaded model from disk") 

        self.root = Tk()
        self.root.title("Character Recogniser")
        
        self.canvas = Canvas(self.root, bg='white', width=280, height=280)
        self.canvas.grid(row=0, columnspan=2)
           
        self.clear_button = Button(self.root, text='Clear', command=self.clear)
        self.clear_button.grid(row=1, column=1)
        
        self.predict_button = Button(self.root, text='Predict!', command=self.predict)
        self.predict_button.grid(row=2, column=1)
        
        self.result_label = Label(self.root, text="Draw something!")
        self.result_label.grid(row=1, column=0)        
        
        self.old_x, self.old_y = None, None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.release)        
        self.root.mainloop()
    
        
    def clear(self):
        self.canvas.delete("all")
        self.result_label['text'] = 'Draw something!'
        global img
        array = np.full((280,280),255)
        img = Image.fromarray(array.astype(np.uint8))
        img.save("output.tif")
        
    def predict(self):
        #global array
        #img = Image.fromarray(array.astype(np.uint8))
        #img.save("output.tif")
        global img
        
        img = img.resize((28, 28),Image.BILINEAR)
        img.save("output.tif")
        #prediction = Averaged_Tester.test_one(darken(img))
        
        #dark_img = darken(img)
        #img_lst = list(dark_img.getdata())
        np_img_lst = np.array(img.getdata())
        
        np_img_lst = (255 - np_img_lst)
        np_img_lst = np_img_lst.reshape(784)
        
        np_img_lst[np_img_lst > 0] = 255
        

        np_img_lst = np_img_lst.reshape(28, 28)
        
        print(np_img_lst.reshape(784))
        
        from matplotlib import pyplot as plt
        plt.imshow(np_img_lst, cmap='gray')

        # normalize data
        np_img_lst = np_img_lst.astype('float32')
        np_img_lst /= 255

        # reshape data
        np_img_lst = np_img_lst.reshape(1, 1, 28, 28)
        print(np_img_lst.shape)

        prediction = self.model.predict(np_img_lst)
        
        # get index with max probability, aka predicted number
        print(prediction.shape)
        max_index = np.argmax(prediction)

        self.result_label['text'] = 'Prediction: ' + str(max_index) + "\n\n" \
            + str(prediction)
                                  
    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, width=self.brush_size, fill='black', capstyle=ROUND)
            
            global img
            draw = ImageDraw.Draw(img)
            draw.line((self.old_x,self.old_y, event.x,event.y), fill=0, width = 1)
            
        self.old_x, self.old_y = event.x, event.y
        
    
    def release(self, event):
        self.old_x, self.old_y = None, None
    
def darken(img):
    
    matrix = Extractor.ImageToMatrix(img)
    #print(matrix)
    #print("===========================")
    matrix.setflags(write=1)
    darkest = 255
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col] < darkest:
                darkest = matrix[row][col]
    # enhace the image
    constant = 255//darkest
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            if matrix[row][col]!=255:
                # Linear Fit                
                matrix[row][col] = 255 - matrix[row][col] * constant
                
                # root Fit
                #a,b = root_fit(darkest)[0], root_fit(darkest)[1]
                #matrix[row][col] = int(math.sqrt((matrix[row][col] + b)/ a))
    #print(matrix)            
    new_img = Image.fromarray(matrix.astype(np.uint8))
    new_img.save("enhanced.tif") 
    return new_img

def root_fit(darkest_value):
    # x = a y^2 + b
    b = -1 * darkest_value
    a = (255+b)/(255*255)
    return [a,b]

def quadratic_fit(darkest_value):
    # y = a x^2 + b
    a = 255/(255*255-darkest_value*darkest_value)
    b = 255-255*255*a
    return [a,b]
                
if __name__ == '__main__':
    Paint()


