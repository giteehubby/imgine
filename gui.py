import tkinter as tk  
from tkinter import filedialog, ttk
from tkinter import messagebox
from i2t import jpg2tensor, tensor2pil
from range_compression import dyrc
from PIL import Image, ImageTk
from ladder import ladder
from threshd import threshold, scared
from hiseq import histo_equa
from conv import denoising, select_mask_smooth, select_mask_smooth_vectorized
from media import media_conv
from prewitt import prewitti, sobel, laplacian


class ImageProcessorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Image Processing Tool - Dark Theme")
        self.root.geometry("1200x800")
        
        # 深色主题颜色配置
        self.colors = {
            'bg_dark': '#1e1e1e',           # 主背景色
            'bg_darker': '#2d2d2d',         # 次背景色
            'bg_light': '#3c3c3c',          # 浅色背景
            'text_primary': '#ffffff',      # 主要文字
            'text_secondary': '#cccccc',    # 次要文字
            'text_muted': '#999999',        # 静音文字
            'accent_blue': '#007acc',       # 蓝色强调
            'accent_green': '#4ec9b0',      # 绿色强调
            'accent_orange': '#ce9178',     # 橙色强调
            'accent_purple': '#c586c0',     # 紫色强调
            'accent_pink': '#dcdcaa',       # 粉色强调
            'border': '#404040',            # 边框色
            'button_hover': '#4a4a4a'       # 按钮悬停色
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        # 全局变量
        self.file_path = None
        self.pil_img = None
        self.original_image_label = None
        self.processed_image_label = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # 主容器
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题
        title_label = tk.Label(main_frame, text="图像处理工具", 
                              font=('Arial', 18, 'bold'), 
                              bg=self.colors['bg_dark'], fg=self.colors['text_primary'])
        title_label.pack(pady=(0, 20))
        
        # 文件操作区域
        self.create_file_section(main_frame)
        
        # 图像显示区域
        self.create_image_section(main_frame)
        
        # 处理按钮区域
        self.create_processing_section(main_frame)
        
    def create_file_section(self, parent):
        file_frame = tk.LabelFrame(parent, text="文件操作", 
                                  font=('Arial', 11, 'bold'),
                                  bg=self.colors['bg_dark'], fg=self.colors['text_primary'],
                                  relief=tk.FLAT, bd=1, highlightbackground=self.colors['border'])
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件路径显示
        self.file_label = tk.Label(file_frame, text="请选择图像文件...", 
                                  bg=self.colors['bg_dark'], fg=self.colors['text_muted'],
                                  font=('Arial', 9))
        self.file_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 按钮区域
        button_frame = tk.Frame(file_frame, bg=self.colors['bg_dark'])
        button_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        select_button = tk.Button(button_frame, text="选择文件", 
                                 command=self.select_file,
                                 bg=self.colors['accent_green'], fg=self.colors['text_primary'],
                                 font=('Arial', 9, 'bold'),
                                 relief=tk.FLAT, bd=0,
                                 activebackground=self.colors['button_hover'],
                                 activeforeground=self.colors['text_primary'])
        select_button.pack(side=tk.LEFT, padx=5)
        
        save_button = tk.Button(button_frame, text="保存文件", 
                               command=self.save_image_as_jpg,
                               bg=self.colors['accent_blue'], fg=self.colors['text_primary'],
                               font=('Arial', 9, 'bold'),
                               relief=tk.FLAT, bd=0,
                               activebackground=self.colors['button_hover'],
                               activeforeground=self.colors['text_primary'])
        save_button.pack(side=tk.LEFT, padx=5)
        
    def create_image_section(self, parent):
        image_frame = tk.Frame(parent, bg=self.colors['bg_dark'])
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # 原始图像区域
        original_frame = tk.LabelFrame(image_frame, text="原始图像", 
                                      font=('Arial', 11, 'bold'),
                                      bg=self.colors['bg_dark'], fg=self.colors['text_primary'],
                                      relief=tk.FLAT, bd=1, highlightbackground=self.colors['border'])
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.original_image_label = tk.Label(original_frame, text="请选择图像文件",
                                            bg=self.colors['bg_darker'], fg=self.colors['text_muted'],
                                            font=('Arial', 12),
                                            relief=tk.FLAT, bd=1, highlightbackground=self.colors['border'])
        self.original_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 处理后图像区域
        processed_frame = tk.LabelFrame(image_frame, text="处理后图像", 
                                       font=('Arial', 11, 'bold'),
                                       bg=self.colors['bg_dark'], fg=self.colors['text_primary'],
                                       relief=tk.FLAT, bd=1, highlightbackground=self.colors['border'])
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.processed_image_label = tk.Label(processed_frame, text="处理后的图像将显示在这里",
                                             bg=self.colors['bg_darker'], fg=self.colors['text_muted'],
                                             font=('Arial', 12),
                                             relief=tk.FLAT, bd=1, highlightbackground=self.colors['border'])
        self.processed_image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def create_processing_section(self, parent):
        processing_frame = tk.LabelFrame(parent, text="图像处理功能", 
                                        font=('Arial', 11, 'bold'),
                                        bg=self.colors['bg_dark'], fg=self.colors['text_primary'],
                                        relief=tk.FLAT, bd=1, highlightbackground=self.colors['border'])
        processing_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 创建按钮网格
        buttons = [
            # 第一行：去噪和滤波
            ("去除孤立噪声点", lambda: self.process_image(denoising), self.colors['accent_orange']),
            ("选择掩码平滑", lambda: self.process_image(select_mask_smooth), self.colors['accent_orange']),
            ("选择掩码平滑(优化)", lambda: self.process_image(select_mask_smooth_vectorized), self.colors['accent_orange']),
            
            # 第二行：滤波和增强
            ("中值滤波", lambda: self.process_image(media_conv), self.colors['accent_orange']),
            ("直方图均衡化", lambda: self.process_image(histo_equa), self.colors['accent_purple']),
            ("动态范围压缩", lambda: self.process_image(dyrc), self.colors['accent_purple']),
            
            # 第三行：增强和锐化
            ("阶梯量化", lambda: self.process_image(lambda x: ladder(x, 60)), self.colors['accent_purple']),
            ("Prewitt锐化", lambda: self.process_image(prewitti), self.colors['accent_pink']),
            ("Sobel锐化", lambda: self.process_image(sobel), self.colors['accent_pink']),
            
            # 第四行：锐化和阈值
            ("Laplacian二阶微分", lambda: self.process_image(laplacian), self.colors['accent_pink']),
            ("阈值切分", lambda: self.process_image(lambda x: threshold(x, 127)), self.colors['accent_blue']),
            ("恐怖风", lambda: self.process_image(lambda x: scared(x)), self.colors['accent_green'])
        ]
        
        # 创建按钮网格
        for i, (text, command, color) in enumerate(buttons):
            row = i // 3
            col = i % 3
            
            button = tk.Button(processing_frame, text=text, command=command,
                              bg=color, fg=self.colors['text_primary'],
                              font=('Arial', 9, 'bold'),
                              relief=tk.FLAT, bd=0,
                              width=18, height=2,
                              activebackground=self.colors['button_hover'],
                              activeforeground=self.colors['text_primary'])
            button.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            
        # 配置网格权重
        for i in range(3):
            processing_frame.grid_columnconfigure(i, weight=1)
            
    def scale_image(self, image, width_limit, height_limit):
        width, height = image.size
        r = max(width//width_limit, height//height_limit)
        if r == 0:
            return image
        new_width = width//r
        new_height = height//r
        return image.resize((new_width, new_height), Image.LANCZOS)
        
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"文件: {file_path.split('/')[-1]}")
            
            # 显示原始图像
            pil_image = self.scale_image(Image.open(file_path), 400, 300)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            self.original_image_label.config(image=tk_image, text="")
            self.original_image_label.image = tk_image
            
            # 清空处理后图像区域
            self.processed_image_label.config(image="", text="处理后的图像将显示在这里")
        else:
            messagebox.showinfo("提示", "未选择文件")
            
    def save_image_as_jpg(self):
        if self.pil_img:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG文件", "*.jpg"), ("PNG文件", "*.png"), ("所有文件", "*.*")]
            )
            if save_path:
                self.pil_img.save(save_path)
                messagebox.showinfo("成功", "图像保存成功！")
        else:
            messagebox.showwarning("警告", "没有可保存的图像！")
            
    def process_image(self, func):
        if not self.file_path:
            messagebox.showwarning("警告", "请先选择图像文件！")
            return
            
        try:
            self.pil_img = tensor2pil(func(jpg2tensor(self.file_path)))
            processed_image = self.scale_image(self.pil_img, 400, 300)
            tk_image = ImageTk.PhotoImage(processed_image)
            
            self.processed_image_label.config(image=tk_image, text="")
            self.processed_image_label.image = tk_image
            
        except Exception as e:
            messagebox.showerror("错误", f"处理图像时出错：{str(e)}")
            
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ImageProcessorGUI()
    app.run()
