import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
# from matplotlib import pyplot as plt

from train import train
from infer import stylize_one
from utils import list_images
from scipy.misc import imread, imsave, imresize
# for training
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'

STYLE_WEIGHTS = [2.0]
MODEL_SAVE_PATHS = [
    'models/style_weight_2e0.ckpt',
]

# for inferring (stylize)
INFERRING_CONTENT_DIR = 'images/Images/content'
INFERRING_STYLE_DIR = 'images/Images/style'

infer_weight = 0.5

class windows:
    def __init__(self, img, clipLimit):
        self.window = tk.Tk()

        self.cur_img1 = img
        self.cur_img2 = img
        self.cur_img3 = img
        self.file_select1 = ''
        self.file_select2 = ''
        self.window.title('imageEditor')
        
        menubar = tk.Menu(self.window)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="open1", command=self.onOpen1)
        filemenu.add_command(label="open2", command=self.onOpen2)
        menubar.add_cascade(label="menu", menu=filemenu)
        self.window.config(menu=menubar)
    
        img_size = 512
        pad = 5
        sticky_mode = 'nswe'

        div1 = tk.Frame(self.window, width=img_size, height=img_size, bg='gray')
        div2 = tk.Frame(self.window, width=img_size, height=img_size, bg='gray')
        div3 = tk.Frame(self.window, width=img_size, height=img_size, bg='gray')
        div4 = tk.Frame(self.window, width=img_size, height=img_size, bg='gray')
        div1.grid(column=0, row=0, padx=pad, pady=pad, rowspan=2, sticky=sticky_mode)
        div2.grid(column=1, row=0, padx=pad, pady=pad, rowspan=2, sticky=sticky_mode)
        div3.grid(column=2, row=0, padx=pad, pady=pad, rowspan=2, sticky=sticky_mode)
        div4.grid(column=3, row=0, padx=pad, pady=pad, sticky='n')

        im_rgb = cv2.cvtColor(self.cur_img1, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        imgTK = ImageTk.PhotoImage(im)

        self.image_label1 = tk.Label(div1, image=imgTK)
        self.image_label1['height'] = img_size
        self.image_label1['width'] = img_size
        self.image_label1.grid(column=0, row=0, sticky=sticky_mode)
        self.image_label1.image = imgTK

        self.image_label2 = tk.Label(div2, image=imgTK)
        self.image_label2['height'] = img_size
        self.image_label2['width'] = img_size
        self.image_label2.grid(column=1, row=0, sticky=sticky_mode)
        self.image_label2.image = imgTK

        self.image_label3 = tk.Label(div3, image=imgTK)
        self.image_label3['height'] = img_size
        self.image_label3['width'] = img_size
        self.image_label3.grid(column=2, row=0, sticky=sticky_mode)
        self.image_label3.image = imgTK

        self.style_weight = tk.Scale(div4, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=clipLimit)
        self.style_button = tk.Button(div4, text='style', bg='gray', fg='white', font=('Arial, 12'))
        self.change_button = tk.Button(div4, text='change', bg='gray', fg='white', font=('Arial, 12'))
        self.save_button = tk.Button(div4, text='save_image', bg='gray', fg='white', font=('Arial, 12'))

        self.style_button['command'] = self.button_event
        self.change_button['command'] = self.change_event
        self.save_button['command'] = self.saveImage

        self.style_weight.grid(column=1, row=0, sticky=sticky_mode)
        self.style_button.grid(column=1, row=2, sticky=sticky_mode)
        self.change_button.grid(column=1, row=3, sticky=sticky_mode)
        self.save_button.grid(column=1, row=4, sticky=sticky_mode)


    def display(self):
        self.window.mainloop()

    def change_event(self):
        self.cur_img1, self.cur_img2 = self.cur_img2, self.cur_img1
        self.file_select1, self.file_select2 = self.file_select2, self.file_select1
        image1 = self.image_label1.image
        image2 = self.image_label2.image
        self.image_label1.configure(image=image2)
        self.image_label1.image = image2
        self.image_label2.configure(image=image1)
        self.image_label2.image = image1


    def button_event(self):
        content_imgs_path = []
        content_imgs_path.append(self.file_select1)
        style_imgs_path = []
        style_imgs_path.append(self.file_select2)
        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):

            print('\n>>> Begin to stylize images with style weight: %.2f\n' % self.style_weight.get())

            img = stylize_one(content_imgs_path, style_imgs_path, 
                    ENCODER_WEIGHTS_PATH, model_save_path, 
                    infer_weight=self.style_weight.get())
            imsave("test.jpg", img)
            print("shape of img: ", np.shape(img))

            print('\n>>> Successfully! Done all stylizing...\n')

        self.cur_img3 = img
        im = Image.fromarray(np.uint8(self.cur_img3))
        imgTK = ImageTk.PhotoImage(im)
        self.image_label3.configure(image=imgTK)
        self.image_label3.image = imgTK

    def saveImage(self):
        file = filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=(("PNG file", "*.png"),("All Files", "*.*") ))
        if file:
            print(os.path.abspath(file.name))
            img_bgr = self.cur_img3[:,:,[2, 1, 0]]
            cv2.imwrite(os.path.abspath(file.name), img_bgr)
        #cv2.imwrite('style{clipLimit}.jpg'.format(clipLimit=self.style_weight.get()), self.cur_img3)

    def onOpen1(self):
        self.file_select1 = (filedialog.askopenfilename(initialdir = "./images/images",title = "Open file",filetypes = (("jpeg File","*.jpg *.png"), ("png File","*.png"))))
        print(self.file_select1)
        self.cur_img1 = cv2.imread(self.file_select1)
        im_rgb = cv2.cvtColor(self.cur_img1, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        imgTK = ImageTk.PhotoImage(im)
        self.image_label1.configure(image=imgTK)
        self.image_label1.image = imgTK

    def onOpen2(self):
        self.file_select2 = (filedialog.askopenfilename(initialdir = "./images/images",title = "Open file",filetypes = (("jpeg File","*.jpg *.png"), ("png File","*.png"))))
        print(self.file_select2)
        self.cur_img1 = cv2.imread(self.file_select2)
        im_rgb = cv2.cvtColor(self.cur_img1, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im_rgb)
        imgTK = ImageTk.PhotoImage(im)
        self.image_label2.configure(image=imgTK)
        self.image_label2.image = imgTK

if __name__ == '__main__':
    # read image
    # bgr = cv2.imread('messala-ciulla-iajsWsOZNWk-unsplash.jpg')
    blank_image = np.zeros((512,512,3), np.uint8)
    
    ui = windows(blank_image, infer_weight)
    ui.display()
