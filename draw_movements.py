import tkinter as tk
import numpy as np

actions = np.genfromtxt("actions.csv")
counter = 0
counts = np.zeros(100)
index = 0
print(actions)
def counter_labels(labels):
  def count():
    global counter
    global index
    global counts
    if counter < len(actions):
        counter += 1
        index = int(actions[counter])
        counts[index] += 1
        label = labels[index]
        label.config(text=str(counts[index]))
        label.after(500, count)
  count()

root = tk.Tk()
root.title("Counting Actions")
nrow, ncol = 10, 10

labels = []
for r in range(nrow):
    for c in range(ncol):
        labels += [tk.Label(root, text="0", borderwidth=1)]
        labels[r * nrow + c].grid(row=r,column=c)

counter_labels(labels)

#label = tk.Label(root, fg="green")
#print(label)
# label.pack()
# counter_label(label)
#button = tk.Button(root, text='Stop', width=25, command=root.destroy)
#button.pack()
root.mainloop()
