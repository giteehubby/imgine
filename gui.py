import tkinter as tk  
from tkinter import filedialog  
from tkinter import messagebox  
  
def select_file():  
    # 打开文件选择对话框  
    file_path = filedialog.askopenfilename()  
      
    # 检查用户是否选择了文件  
    if file_path:  
        # 在标签中显示文件路径  
        file_label.config(text=f"选中的文件: {file_path}")  
    else:  
        # 显示消息框，用户取消了选择  
        messagebox.showinfo("信息", "没有选择文件")  
  
# 创建主窗口  
root = tk.Tk()  
root.title("文件选择器示例")  
  
# 创建一个标签来显示文件路径  
file_label = tk.Label(root, text="请选择文件...")  
file_label.pack(pady=20)  
  
# 创建一个按钮来触发文件选择对话框  
select_button = tk.Button(root, text="选择文件", command=select_file)  
select_button.pack(pady=20)  
  
# 运行主循环  
root.mainloop()