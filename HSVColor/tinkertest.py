import tkinter as tk

root = tk.Tk()
root.geometry("400x300")

hue_min = tk.IntVar(value=0)
tk.Label(root, text="Hue Min").pack()
tk.Scale(root, from_=0, to=179, orient=tk.HORIZONTAL, variable=hue_min).pack()

root.mainloop()