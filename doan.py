import cv2
import pytesseract
from tkinter import *
from PIL import ImageGrab
from playsound import  playsound
import os
def Nhandien(path):
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    #
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # print(pytesseract.image_to_string(img))
    print(pytesseract.image_to_boxes(img))

    ## Tìm kiếm ký tự
    hImg, wImg, _ = img.shape
    boxes = pytesseract.image_to_boxes(img)
    res = pytesseract.image_to_string(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(img, (x,hImg-y), (w, hImg-h), (0,0,255),3)
        cv2.putText(img, b[0], (x,hImg-y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 2)

    # cv2.imshow('Result', img)
    # cv2.waitKey(0)
    return res.split('\n')[0]

def getter(widget):
    x=root.winfo_rootx()+widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()
    ImageGrab.grab().crop((x,y,x1,y1)).save("test.jpg")

def HandleListen():
    getter(wn)
    res = Nhandien('test.jpg')
    p = './sound/' + res.lower() + '.mp3'
    playsound(p)

def HandleClick():
    getter(wn)
    res = Nhandien('test.jpg')
    if res != None:
        result['text'] = res
    else:
        result['text'] = 'error'
    print(result['text'])
def HandleImage():
    getter(wn)
    res = Nhandien('test.jpg')
    path = './img/' + res.lower() + '.jpg'
    img = cv2.imread(path)
    width = int(img.shape[1] * 60 / 100)
    height = int(img.shape[0] * 60 / 100)
    resized = cv2.resize(img, (width,height))
    cv2.imshow('Result', resized)

def HandleClear():
    wn.delete("all")

root = Tk()
root.title("Paint Application")
root.geometry("800x600")
def paint(event):
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-3), (event.y-3)
    x2, y2 = (event.x+3), (event.y+3)
    color = "black"
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color)
# create canvas
wn = Canvas(root, width=700, height=500, bg='white')
# bind mouse event with canvas(wn)
wn.bind('<B1-Motion>', paint)
wn.pack()
# create label
result = Label(root, text='Hello World', font=("Helvetica", 15))
result.pack(padx=10)
# create button
frame = Frame(root)
frame.pack(side=BOTTOM)
btnShow = Button(frame, text='Show', command=HandleClick)
btnShow.pack(side=LEFT, padx=20)
btnListen = Button(frame, text='Listen', command=HandleListen)
btnListen.pack(side=LEFT, padx=20)
btnImage = Button(frame, text='Image', command=HandleImage)
btnImage.pack(side=LEFT, padx=20)
btnClear = Button(frame, text='Clear', command=HandleClear)
btnClear.pack(side=LEFT, padx=20)
root.mainloop()

