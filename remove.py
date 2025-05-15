import tkinter as tk
from tkinter import filedialog

def remove_comments(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    comments_removed = []
    for line in lines:
        if line.strip().startswith('#'):
            continue
        comments_removed.append(line)
    
    with open(file_path, 'w') as file:
        file.writelines(comments_removed)

def select_file():
    file_path = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def process_file():
    file_path = entry.get()
    if file_path:
        remove_comments(file_path)
        label.config(text="Comments removed successfully!")

root = tk.Tk()
root.title("Comment Remover")

label = tk.Label(root, text="Select a Python file:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

button = tk.Button(root, text="Browse", command=select_file)
button.pack()

button = tk.Button(root, text="Remove Comments", command=process_file)
button.pack()

label = tk.Label(root, text="")
label.pack()

root.mainloop()