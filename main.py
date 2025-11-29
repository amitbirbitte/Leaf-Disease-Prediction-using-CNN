from tkinter import *
from tkinter import messagebox


from PIL import ImageTk, Image
import sqlite3
import os
root = Tk()
root.geometry('390x600')
root.title("Raitha")

canv = Canvas(root, width=1366, height=768, bg='white')
canv.grid(row=2, column=3)
img = Image.open('back.png')
photo = ImageTk.PhotoImage(img)
canv.create_image(1,1, anchor=NW, image=photo)
def readimg():
    os.system('python readimg.py')
def preprocessing():
    os.system('python preprocessing.py')
def cnn():
    os.system('python cnn.py')
def seg():
        os.system('python seg.py')


def det1():
    os.system('python perdet.py')
def captimg():
    os.system('python captimg.py')
Button(root, text='Capture Image', width=20, bg='yellow', fg='black',  font=("bold", 8),command=captimg).place(x=100, y=250)
Button(root, text='Read Image', width=20, bg='yellow', fg='black',  font=("bold", 8),command=readimg).place(x=100, y=300)
Button(root, text='Preprocessing', width=20, bg='yellow', fg='black',  font=("bold", 8),command=preprocessing).place(x=100, y=330)
Button(root, text='Segmentation', width=20, bg='yellow', fg='black',  font=("bold", 8),command=seg).place(x=100, y=360)
Button(root, text='CNN Model', width=20, bg='yellow', fg='black',  font=("bold", 8),command=cnn).place(x=100, y=390)
Button(root, text='Predict', width=20, bg='yellow', fg='black', command=det1, font=("bold", 8)).place(x=100, y=420)

root.mainloop()
