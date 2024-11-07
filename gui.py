import tkinter as tk  
from tkinter import filedialog  
from tkinter import messagebox
from i2t import jpg2tensor, tensor2pil
from range_compression import dyrc
from PIL import Image, ImageTk
from ladder import ladder
from threshd import threshold
from hiseq import histo_equa
from conv import denoising,select_mask_smooth
from media import media_conv
from prewitt import prewitti, sobel, laplacian


def config_adapt(num_grid_col,num_grid_row):
    for col in range(num_grid_col):
        root.grid_columnconfigure(col, weight=1)
    for row in range(num_grid_row):
        root.grid_rowconfigure(row,weight=1)

def scale(image, width_limit,height_limit):
    width, height = image.size
    r = max(width//width_limit, height//height_limit)
    if r == 0:
        return image
    new_width = width//r
    new_height = height//r
    return image.resize((new_width,new_height),Image.LANCZOS)

def select_file():  
    # 打开文件选择对话框
    global file_path
    file_path = filedialog.askopenfilename()  
      
    # 检查用户是否选择了文件  
    if file_path:  
        # 在标签中显示文件路径
        file_label.config(text=f"选中的文件: {file_path}")
        pil_image = scale(Image.open(file_path),400,300)
        tk_image = ImageTk.PhotoImage(pil_image)
        label = tk.Label(root, image=tk_image,borderwidth=5)
        label.image = tk_image  # 保持对PhotoImage的引用，防止被垃圾回收
        label.grid(column=0,row=1,columnspan=1,rowspan=1)

    else:  
        # 显示消息框，用户取消了选择  
        messagebox.showinfo("信息", "没有选择文件")

def save_image_as_jpg():  
    if 'pil_img' in globals() and pil_img:  
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])  
        if save_path:  
            pil_img.save(save_path, "JPEG")  
            messagebox.showinfo("Success", "Image saved successfully!")  
    else:  
        messagebox.showwarning("Warning", "No image to save!")  


def fuction_button(f):
    global pil_img
    pil_img = tensor2pil(f(jpg2tensor(file_path)))
    tk_image_2 = ImageTk.PhotoImage(
        scale(pil_img,400,300)
    )
    label_2 = tk.Label(root,image=tk_image_2,borderwidth=5)
    label_2.image = tk_image_2
    label_2.grid(column=1,columnspan=1,row=1,rowspan=1)

# 创建主窗口  
root = tk.Tk()  
root.title("imagine")
root.geometry("1000x700")
frame = tk.Frame(root, bg='black')  
frame.grid(row=0, column=0, rowspan=5,columnspan=4,sticky=tk.W+tk.E+tk.N+tk.S)



# 创建一个标签来显示文件路径
file_label = tk.Label(root, text="请选择文件...")  
file_label.grid(row=4,column=0,columnspan=3,pady=20)

# 创建一个按钮来触发文件选择对话框  
select_button = tk.Button(root, text="选择文件", command=select_file)  
select_button.grid(row=3,column=0,pady=20)

save_button = tk.Button(root, text="保存文件", command=save_image_as_jpg)  
save_button.grid(row=3,column=1,pady=20)


dyrc_button = tk.Button(root, text="动态范围压缩", command=lambda:fuction_button(dyrc))
dyrc_button.grid(row=2,column=2,pady=20)

ladder_button = tk.Button(root, text="阶梯量化", command=lambda:fuction_button(lambda x:ladder(x,60)))
ladder_button.grid(row=3,column=2,pady=20)

thres_button = tk.Button(root, text="阈值切分", command=lambda:fuction_button(lambda x:threshold(x,127)))
thres_button.grid(row=4,column=2,pady=20)

hiseq_button = tk.Button(root, text="直方图均衡化", command=lambda:fuction_button(histo_equa))
hiseq_button.grid(row=1,column=2,pady=20)

denois_button = tk.Button(root, text="去除孤立噪声点", command=lambda:fuction_button(denoising))
denois_button.grid(row=0,column=2,pady=20)

sms_button = tk.Button(root, text="选择掩码平滑", command=lambda:fuction_button(select_mask_smooth))
sms_button.grid(row=0,column=3,pady=20)

media_button = tk.Button(root, text="中值滤波", command=lambda:fuction_button(media_conv))
media_button.grid(row=1,column=3,pady=20)

pt_button = tk.Button(root, text="prewitt锐化", command=lambda:fuction_button(prewitti))
pt_button.grid(row=2,column=3,pady=20)

sb_button = tk.Button(root, text="sobel锐化", command=lambda:fuction_button(sobel))
sb_button.grid(row=3,column=3,pady=20)

lp_button = tk.Button(root, text="laplacian二阶微分", command=lambda:fuction_button(laplacian))
lp_button.grid(row=4,column=3,pady=20)




config_adapt(4,4)
  
# 运行主循环  
root.mainloop()
