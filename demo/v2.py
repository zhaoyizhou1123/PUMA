import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

class SudokuVisualizer:
    def __init__(self, root, data=None):
        self.root = root
        self.root.title("Sudoku Generation Visualizer")
        
        # Keep track of the full (N, T, seq_len) dataset
        self.full_data = None
        self.current_idx = 0
        
        # Initialize with provided data or a blank 81x81 template
        self.data = data if data is not None else np.zeros((81, 81), dtype=int)
        self.step = 0
        self.max_steps = max(0, self.data.shape[0] - 1)
        
        self.cells = {}
        self.create_grid()
        self.create_controls()
        
        self.update_grid()

    def create_grid(self):
        grid_frame = tk.Frame(self.root, bg="black")
        grid_frame.pack(padx=20, pady=20)
        
        for i in range(9):
            for j in range(9):
                pad_x = (1, 3) if j % 3 == 2 and j != 8 else (1, 1)
                pad_y = (1, 3) if i % 3 == 2 and i != 8 else (1, 1)
                
                frame = tk.Frame(grid_frame, bg="white", width=50, height=50)
                frame.grid(row=i, column=j, padx=pad_x, pady=pad_y)
                frame.pack_propagate(False)
                
                label = tk.Label(frame, text="", font=("Helvetica", 20, "bold"), bg="white")
                label.pack(expand=True, fill="both")
                self.cells[(i, j)] = label

    def create_controls(self):
        # --- ROW 1: File and Index Controls ---
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=5)
        
        self.load_btn = tk.Button(top_frame, text="Load .npy", command=self.load_file, width=10, bg="#e0e0e0")
        self.load_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="  |  Sample Index:").pack(side=tk.LEFT)
        self.index_entry = tk.Entry(top_frame, width=5)
        self.index_entry.pack(side=tk.LEFT, padx=5)
        self.index_entry.insert(0, "0")
        
        self.index_max_label = tk.Label(top_frame, text="(Max: 0)")
        self.index_max_label.pack(side=tk.LEFT)
        
        self.set_index_btn = tk.Button(top_frame, text="Set Index", command=self.set_index)
        self.set_index_btn.pack(side=tk.LEFT, padx=5)

        # --- ROW 2: Playback Controls ---
        playback_frame = tk.Frame(self.root)
        playback_frame.pack(pady=5)

        self.prev_btn = tk.Button(playback_frame, text="< Prev", command=self.prev_step, width=8)
        self.prev_btn.pack(side=tk.LEFT, padx=10)
        
        self.play_btn = tk.Button(playback_frame, text="Play", command=self.play, width=8)
        self.play_btn.pack(side=tk.LEFT, padx=10)
        
        self.next_btn = tk.Button(playback_frame, text="Next >", command=self.next_step, width=8)
        self.next_btn.pack(side=tk.LEFT, padx=10)

        # --- ROW 3: Step Jump Controls ---
        step_frame = tk.Frame(self.root)
        step_frame.pack(pady=5)

        self.step_label = tk.Label(step_frame, text=f"Step: {self.step}/{self.max_steps}", font=("Helvetica", 12))
        self.step_label.pack(side=tk.LEFT, padx=10)
        
        tk.Label(step_frame, text="  Jump to:").pack(side=tk.LEFT)
        self.step_entry = tk.Entry(step_frame, width=5)
        self.step_entry.pack(side=tk.LEFT, padx=5)
        
        self.jump_btn = tk.Button(step_frame, text="Go", command=self.jump_to_step)
        self.jump_btn.pack(side=tk.LEFT, padx=5)

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Sudoku Numpy File",
            filetypes=(("Numpy Files", "*.npy"), ("All Files", "*.*"))
        )
        
        if not filepath:
            return 
            
        try:
            raw_data = np.load(filepath)
            self.full_data = np.transpose(raw_data, (1, 0, 2)) # (N, T, seq_len)
            print("New data shape:", self.full_data.shape)

            # Update Max Index Label
            max_idx = self.full_data.shape[0] - 1
            self.index_max_label.config(text=f"(Max: {max_idx})")
            
            # Reset index entry to 0
            self.index_entry.delete(0, tk.END)
            self.index_entry.insert(0, "0")
            
            # Update window title
            filename = filepath.split('/')[-1]
            self.root.title(f"Sudoku Visualizer - {filename}")
            
            # Automatically load index 0
            self.set_index()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def set_index(self):
        if self.full_data is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return
            
        try:
            target_idx = int(self.index_entry.get())
            max_idx = self.full_data.shape[0] - 1
            
            if 0 <= target_idx <= max_idx:
                self.current_idx = target_idx
                current_sol = self.full_data[self.current_idx, :, 81:] # (T, 81)
                
                if current_sol.shape[1] != 81:
                    messagebox.showerror("Error", f"Expected inner shape (N, 81), got {current_sol.shape}")
                    return
                
                self.data = current_sol
                self.step = 0
                self.max_steps = self.data.shape[0] - 1
                self.update_grid()
            else:
                messagebox.showerror("Error", f"Index out of bounds! Must be between 0 and {max_idx}.")
                self.index_entry.delete(0, tk.END)
                self.index_entry.insert(0, str(self.current_idx))
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for the index.")

    def jump_to_step(self):
        try:
            target_step = int(self.step_entry.get())
            if 0 <= target_step <= self.max_steps:
                self.step = target_step
                self.update_grid()
            else:
                messagebox.showerror("Error", f"Step out of bounds! Must be between 0 and {self.max_steps}.")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for the step.")
            
        # Clear the entry box after jumping
        self.step_entry.delete(0, tk.END)

    def update_grid(self):
        current_state = self.data[self.step].reshape(9, 9)
        prev_state = self.data[self.step - 1].reshape(9, 9) if self.step > 0 else np.zeros((9, 9))
        
        for i in range(9):
            for j in range(9):
                val = current_state[i, j]
                text = str(val) if val != 0 else ""
                
                self.cells[(i, j)].config(text=text)
                
                if val != 0 and prev_state[i, j] == 0:
                    self.cells[(i, j)].config(fg="#0052cc") 
                else:
                    self.cells[(i, j)].config(fg="black")
                    
        self.step_label.config(text=f"Step: {self.step}/{self.max_steps}")

    def next_step(self):
        if self.step < self.max_steps:
            self.step += 1
            self.update_grid()

    def prev_step(self):
        if self.step > 0:
            self.step -= 1
            self.update_grid()

    def play(self):
        if self.step < self.max_steps:
            self.next_step()
            self.root.after(150, self.play)


if __name__ == "__main__":
    root = tk.Tk()
    root.eval('tk::PlaceWindow . center')
    app = SudokuVisualizer(root) 
    root.mainloop()