from cx_Freeze import setup, Executable  

setup(  
    name = "d:/imgine/imgine/gui",  
    version = "0.1",  
    description = "My Python Script",  
    executables = [Executable("gui.py")]  
)