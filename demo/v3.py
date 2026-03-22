# Visualize prompt and generation

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import torch

class SudokuVisualizer:
    def __init__(self, root, data=None):
        self.root = root
        self.root.title("Sudoku Generation Visualizer")
        
        # Keep track of the full (N, T, seq_len) dataset
        self.full_data = None
        self.current_idx = 0
        
        # Initialize with blank templates (T, 81)
        self.data_draft = np.zeros((1, 81), dtype=int)
        self.data_output = np.zeros((1, 81), dtype=int)
        
        self.step = 0
        self.max_steps = 0
        
        self.cells_draft = {}
        self.cells_output = {}
        
        self.create_grids()
        self.create_controls()
        
        self.update_grid()

    def create_grids(self):
        # Container for both boards
        boards_container = tk.Frame(self.root)
        boards_container.pack(padx=20, pady=10)

        # --- Draft Board ---
        draft_container = tk.Frame(boards_container)
        draft_container.pack(side=tk.LEFT, padx=20)
        
        tk.Label(draft_container, text="Draft", font=("Helvetica", 16, "bold")).pack(pady=5)
        
        draft_grid = tk.Frame(draft_container, bg="black")
        draft_grid.pack()
        self._build_9x9_grid(draft_grid, self.cells_draft)

        # --- Output Board ---
        output_container = tk.Frame(boards_container)
        output_container.pack(side=tk.LEFT, padx=20)
        
        tk.Label(output_container, text="Output", font=("Helvetica", 16, "bold")).pack(pady=5)
        
        output_grid = tk.Frame(output_container, bg="black")
        output_grid.pack()
        self._build_9x9_grid(output_grid, self.cells_output)

    def _build_9x9_grid(self, parent_frame, cell_dict):
        """Helper to construct a single 9x9 grid inside a given parent frame."""
        for i in range(9):
            for j in range(9):
                pad_x = (1, 3) if j % 3 == 2 and j != 8 else (1, 1)
                pad_y = (1, 3) if i % 3 == 2 and i != 8 else (1, 1)
                
                frame = tk.Frame(parent_frame, bg="white", width=40, height=40)
                frame.grid(row=i, column=j, padx=pad_x, pady=pad_y)
                frame.pack_propagate(False)
                
                label = tk.Label(frame, text="", font=("Helvetica", 18, "bold"), bg="white")
                label.pack(expand=True, fill="both")
                cell_dict[(i, j)] = label

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
        step_frame.pack(pady=10)

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
                
                # Split seq_len into draft (first 81) and output (next 81)
                self.data_draft = self.full_data[self.current_idx, :, :81]
                self.data_output = self.full_data[self.current_idx, :, 81:162]
                
                if self.data_draft.shape[1] < 81 or self.data_output.shape[1] < 81:
                    messagebox.showerror("Error", f"Expected inner shape to have >=162 elements for two grids, got {self.full_data.shape[-1]}")
                    return

                self.verify_sudoku(self.current_idx)  # Print verification results in console
                
                self.step = 0
                self.max_steps = self.data_draft.shape[0] - 1
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
        # Update both Draft and Output grids
        self._update_single_grid(self.data_draft, self.cells_draft)
        self._update_single_grid(self.data_output, self.cells_output)
        
        self.step_label.config(text=f"Step: {self.step}/{self.max_steps}")

    def _update_single_grid(self, data_source, cell_dict):
        current_state = data_source[self.step].reshape(9, 9)
        prev_state = data_source[self.step - 1].reshape(9, 9) if self.step > 0 else np.zeros((9, 9))
        
        for i in range(9):
            for j in range(9):
                val = current_state[i, j]
                if val == 0:
                    text = ""
                elif val == 10:
                    text = "?"
                else:
                    text = str(val)
                
                cell_dict[(i, j)].config(text=text)
                
                if val != 0 and prev_state[i, j] == 0:
                    cell_dict[(i, j)].config(fg="#0052cc") 
                else:
                    cell_dict[(i, j)].config(fg="black")

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

    def verify_sudoku(self, index: int) -> torch.Tensor:
        """
        pred: [B, 162] where pred[:, :81] are clues/condition, pred[:, 81:] is predicted solution
        target: [B, 162] where target[:, :81] are clues/condition, target[:, 81:] is ground-truth solution
        returns: [B] bool
        """
        pred = self.full_data[index:index+1, -1]  # Get the final step prediction for the given index
        orig = self.full_data[index:index+1, 0]   # Get the original clues for the given index
        cond = orig[:, :81]
        sol  = pred[:, 81:]
        print("Cond: ", cond.reshape(9, 9))
        print("Sol: ", sol.reshape(9, 9))

        clue_ok = ((cond == 0) | (sol == cond)).all(axis=1)   # [B]
        sudoku_ok = sudoku_check(torch.tensor(sol)).numpy()                        # [B]

        print("Index:", index)
        print("Clue OK:", clue_ok)
        print("Sudoku OK:", sudoku_ok)
        
        return clue_ok & sudoku_ok 

def sudoku_check(pred: torch.Tensor) -> torch.Tensor:
    """
    Check if the predicted Sudoku solution is valid.
    pred: [B, 81], returns [B] bool
    """
    B, _ = pred.shape
    x = pred.view(B, 9, 9)

    # Must be integers in {1,...,9} (no zeros allowed in a completed Sudoku)
    in_range = (x >= 1) & (x <= 9)

    # Helper: check each length-9 group is a permutation of 1..9
    ref = torch.arange(1, 10, device=pred.device, dtype=pred.dtype).view(1, 1, 9)

    def groups_ok(groups: torch.Tensor) -> torch.Tensor:
        # groups: [B, G, 9]
        sorted_groups, _ = torch.sort(groups, dim=-1)
        return (sorted_groups == ref).all(dim=-1)  # [B, G] bool

    # Rows: [B, 9, 9]
    rows_ok = groups_ok(x)

    # Cols: [B, 9, 9]
    cols_ok = groups_ok(x.transpose(1, 2))

    # 3x3 blocks: reshape into 9 blocks of 9
    blocks = x.view(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).contiguous().view(B, 9, 9)
    blocks_ok = groups_ok(blocks)

    # All constraints must hold + all entries in range
    return in_range.all(dim=(1, 2)) & rows_ok.all(dim=1) & cols_ok.all(dim=1) & blocks_ok.all(dim=1)


if __name__ == "__main__":
    root = tk.Tk()
    root.eval('tk::PlaceWindow . center')
    app = SudokuVisualizer(root) 
    root.mainloop()