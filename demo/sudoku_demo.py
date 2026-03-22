import tkinter as tk
import numpy as np
import torch

class SudokuVisualizer:
    def __init__(self, root, data):
        self.root = root
        self.root.title("Sudoku Generation Visualizer")
        
        # Expects a numpy array of shape (81, 81)
        self.data = data 
        self.step = 0
        self.max_steps = data.shape[0] - 1
        
        self.cells = {}
        self.create_grid()
        self.create_controls()
        
        # Initialize the first frame
        self.update_grid()

    def create_grid(self):
        # Background color black acts as the grid lines
        grid_frame = tk.Frame(self.root, bg="black")
        grid_frame.pack(padx=20, pady=20)
        
        for i in range(9):
            for j in range(9):
                # Add thicker padding to separate the 3x3 blocks
                pad_x = (1, 3) if j % 3 == 2 and j != 8 else (1, 1)
                pad_y = (1, 3) if i % 3 == 2 and i != 8 else (1, 1)
                
                # The individual cell frame
                frame = tk.Frame(grid_frame, bg="white", width=50, height=50)
                frame.grid(row=i, column=j, padx=pad_x, pady=pad_y)
                frame.pack_propagate(False) # Prevent frame from shrinking to label size
                
                # The text label for the number
                label = tk.Label(frame, text="", font=("Helvetica", 20, "bold"), bg="white")
                label.pack(expand=True, fill="both")
                self.cells[(i, j)] = label

    def create_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        self.prev_btn = tk.Button(control_frame, text="< Prev", command=self.prev_step, width=8)
        self.prev_btn.pack(side=tk.LEFT, padx=10)
        
        self.play_btn = tk.Button(control_frame, text="Play", command=self.play, width=8)
        self.play_btn.pack(side=tk.LEFT, padx=10)
        
        self.next_btn = tk.Button(control_frame, text="Next >", command=self.next_step, width=8)
        self.next_btn.pack(side=tk.LEFT, padx=10)

        self.step_label = tk.Label(self.root, text=f"Step: {self.step}/{self.max_steps}", font=("Helvetica", 12))
        self.step_label.pack(pady=5)

    def update_grid(self):
        # Reshape the 81-length 1D array from the current step into a 9x9 2D array
        current_state = self.data[self.step].reshape(9, 9)
        
        # Fetch the previous state to highlight what changed
        prev_state = self.data[self.step - 1].reshape(9, 9) if self.step > 0 else np.zeros((9, 9))
        
        for i in range(9):
            for j in range(9):
                val = current_state[i, j]
                text = str(val) if val != 0 else ""
                
                # Update label text
                self.cells[(i, j)].config(text=text)
                
                # Highlight the newly placed token in blue
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
            # 150ms delay between steps. Adjust this to speed up/slow down.
            self.root.after(150, self.play)

def verify_sudoku(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    pred: [B, 162] where pred[:, :81] are clues/condition, pred[:, 81:] is predicted solution
    target: [B, 162] where target[:, :81] are clues/condition, target[:, 81:] is ground-truth solution
    returns: [B] bool
    """
    cond = pred[:, :81]
    sol  = pred[:, 81:]

    clue_ok = ((cond == 0) | (sol == cond)).any(dim=1)   # [B]
    sudoku_ok = sudoku_check(sol)                        # [B]

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

def visualize_sudoku(puzzle):
    # 9x9 np.array
    def format_cell(x):
        return str(x) if x != 10 else "M"
    for row in puzzle:
        print(" ".join(format_cell(x) for x in row))

if __name__ == "__main__":
    data_file = "data/sudoku_hard/test_mdm.npy"

    data = np.load(data_file) # (N, seq_len)
    print("Data shape:", data.shape)

    sol_file = "track/sudoku_hard-pretraining/progressive_edit-s2028/2026-03-13_15-50-37/step0_rank0.npy"
    sol = np.load(sol_file) # (T, N, seq_len)
    sol = np.transpose(sol, (1, 0, 2)) # (N, T, seq_len)
    print("Solution shape:", sol.shape)

    idx = 0
    current_solution = sol[idx, :, 81:] # (T, 81)

    d = data[idx]
    gt = d[81:]
    final = current_solution[-1]
    correct = np.all(final == gt)

    score = verify_sudoku(torch.tensor(sol[idx, -1:, :]), torch.tensor(d[None, :81]))
    print(f"Matches ground truth: {correct}, Score: {score.item()}")

    question = d[:81].reshape(9,9)
    print("Question:")
    visualize_sudoku(question)

    # Launch GUI
    root = tk.Tk()
    
    # Optional: Center the window
    root.eval('tk::PlaceWindow . center')
    
    # Pass your actual (81,81) array here instead of mock_data
    app = SudokuVisualizer(root, current_solution) 
    
    root.mainloop()