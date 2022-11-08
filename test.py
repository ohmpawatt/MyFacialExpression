import tkinter as tk
import tkinter.filedialog
from tkinter import ttk

from PIL import Image, ImageTk


# 选择并显示图片
def choosepic1():
    path_ = tkinter.filedialog.askopenfilename()
    # path.set(path_)
    
    img_open = Image.open(path_)
    img = ImageTk.PhotoImage(img_open.resize((400, 400)))
    # img = ImageTk.PhotoImage(img_open)
    lableShowImage.config(image=img)
    lableShowImage.image = img

def choosepic2():
    global img0
    photo = Image.open("images/results/l.png")  # 括号里为需要显示在图形化界面里的图片

    photo = photo.resize((1500,400))  # 规定图片大小
    img0 = ImageTk.PhotoImage(photo)
    img1 = ttk.Label(text="照片:", image=img0)
    img1.pack()


if __name__ == '__main__':
    # 生成tk界面 app即主窗口
    app = tk.Tk()
    # 修改窗口titile
    app.title("显示图片")
    # 设置主窗口的大小和位置
    app.geometry("2000x2000")
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
    # 显示结果
    anniu = tk.Button(app, text="点击", command=choosepic2)
    anniu.pack()
    # buttonSelImage.pack(side=tk.BOTTOM)
    # Call the mainloop of Tk.
    app.mainloop()
    # app.mainloop()
