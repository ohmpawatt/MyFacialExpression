"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk

from PIL import Image, ImageTk

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# 选择并显示图片
def choosepic1():
    global path_
    path_ = tkinter.filedialog.askopenfilename()
    # path.set(path_)
    print(path_)
    img_open = Image.open(path_)
    img = ImageTk.PhotoImage(img_open.resize((400, 400)))
    # img = ImageTk.PhotoImage(img_open)
    lableShowImage.config(image=img)
    lableShowImage.image = img
    lableShowImage.place(x=100, y=400)

    raw_img = io.imread(path_)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    with torch.no_grad():
        inputs = Variable(inputs)
    outputs = net(inputs)
    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

    score = F.softmax(outputs_avg, dim=-1)
    _, predicted = torch.max(outputs_avg.data, 0)

    plt.rcParams['figure.figsize'] = (13.5, 5.5)
    axes = plt.subplot(1, 3, 1)
    plt.imshow(raw_img)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 3, 2)
    ind = 0.1 + 0.6 * np.arange(len(class_names))  # the x locations for the groups
    width = 0.4  # the width of the bars: can also be len(x) sequence
    color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title("Classification results ", fontsize=20)
    plt.xlabel(" Expression Category ", fontsize=16)
    plt.ylabel(" Classification Score ", fontsize=16)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)

    axes = plt.subplot(1, 3, 3)
    emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
    plt.imshow(emojis_img)
    plt.xlabel('Emoji Expression', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()
    # show emojis
    plt.savefig(os.path.join('images/results/l.png'))
    # plt.show()
    plt.close()

    print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))


def choosepic2():

    # global img0
    # photo = Image.open("images/results/l.png")  # 括号里为需要显示在图形化界面里的图片
    #
    # photo = photo.resize((1500, 400))  # 规定图片大小
    # img0 = ImageTk.PhotoImage(photo)
    # img1 = ttk.Label(image=img0)
    # img1.pack()
    img_open = Image.open("images/results/l.png")
    img = ImageTk.PhotoImage(img_open.resize((800, 400)))
    # img = ImageTk.PhotoImage(img_open)
    lableShowImage.config(image=img)
    lableShowImage.image = img
    lableShowImage.place(x=500, y=400)


app = tk.Tk()
# 修改窗口titile
app.title("人脸表情识别")
# 设置主窗口的大小和位置
app.geometry("1300x1000")
l=tk.Label(app,text="人脸表情识别系统",font=("Arial",20),width=60,height=5)
l.pack()
# Entry widget which allows displaying simple text.
path = tk.StringVar()
entry = tk.Entry(app, state='readonly', text=path, width=100)
entry.pack()
# 使用Label显示图片
lableShowImage = tk.Label(app)
lableShowImage.pack()
# 选择图片的按钮
buttonSelImage = tk.Button(app, text='选择图片', command=choosepic1)
buttonSelImage.pack()
buttonSelImage.place(x=200, y=300)

# raw_img = io.imread(path_)
# gray = rgb2gray(raw_img)
# gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
#
# img = gray[:, :, np.newaxis]
#
# img = np.concatenate((img, img, img), axis=2)
# img = Image.fromarray(img)
# inputs = transform_test(img)
#
# class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#
# net = VGG('VGG19')
# checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
# net.load_state_dict(checkpoint['net'])
# net.cuda()
# net.eval()
#
# ncrops, c, h, w = np.shape(inputs)
#
# inputs = inputs.view(-1, c, h, w)
# inputs = inputs.cuda()
# with torch.no_grad():
#     inputs = Variable(inputs)
# outputs = net(inputs)
# outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
#
# score = F.softmax(outputs_avg, dim=-1)
# _, predicted = torch.max(outputs_avg.data, 0)
#
# plt.rcParams['figure.figsize'] = (13.5, 5.5)
# axes = plt.subplot(1, 3, 1)
# plt.imshow(raw_img)
# plt.xlabel('Input Image', fontsize=16)
# axes.set_xticks([])
# axes.set_yticks([])
# plt.tight_layout()
#
# plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)
#
# plt.subplot(1, 3, 2)
# ind = 0.1 + 0.6 * np.arange(len(class_names))  # the x locations for the groups
# width = 0.4  # the width of the bars: can also be len(x) sequence
# color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
# for i in range(len(class_names)):
#     plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
# plt.title("Classification results ", fontsize=20)
# plt.xlabel(" Expression Category ", fontsize=16)
# plt.ylabel(" Classification Score ", fontsize=16)
# plt.xticks(ind, class_names, rotation=45, fontsize=14)
#
# axes = plt.subplot(1, 3, 3)
# emojis_img = io.imread('images/emojis/%s.png' % str(class_names[int(predicted.cpu().numpy())]))
# plt.imshow(emojis_img)
# plt.xlabel('Emoji Expression', fontsize=16)
# axes.set_xticks([])
# axes.set_yticks([])
# plt.tight_layout()
# # show emojis
# plt.savefig(os.path.join('images/results/l.png'))
# plt.show()
# plt.close()
#
# print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))

# 显示结果
anniu = tk.Button(app, text="显示结果", command=choosepic2)
anniu.pack()
anniu.place(x=1050, y=300)
# buttonSelImage.pack(side=tk.BOTTOM)
# Call the mainloop of Tk.
app.mainloop()
