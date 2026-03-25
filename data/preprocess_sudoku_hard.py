import pandas as pd
import numpy as np

def filter():
    data_file = "/home/zhaoyiz/projects/PUMA/data/sudoku_hard/train.csv"
    df = pd.read_csv(data_file)

    # select rows with rating > 10
    df_hard = df[df["rating"] > 10]

    save_file = "data/sudoku_hard/hard.csv"
    df_hard.to_csv(save_file, index=False)

def reformat():
    data_file = "data/sudoku_hard/hard.csv"
    df = pd.read_csv(data_file, usecols=["question", "answer"])

    def get_seq(row):
        q = row["question"]
        a = str(row["answer"])
        q = q.replace('.', '0')
        # split into 81 characters, and convert to int
        q = [int(x) for x in q]
        a = [int(x) for x in a]
        return q + a

    df["seq"] = df.apply(get_seq, axis=1)
    # return seq as numpy array
    seqs = df["seq"].tolist()
    # print(seqs)
    # convert into numpy array of shape (N, seq_len)
    seqs = np.array(seqs)
    # print(seqs)
    print(type(seqs[0]))
    save_file = "data/sudoku_hard/hard.npy"
    np.save(save_file, seqs)

def split():
    data_file = "data/sudoku_hard/hard.npy"
    data = np.load(data_file, allow_pickle=True) # (N, seq_len)
    print("Data shape:", data.shape)

    # print(data[0])
    num_test = 1000
    train_data = data[:-num_test]
    test_data = data[-num_test:]

    np.save("data/train_mdm.npy", train_data)
    np.save("data/test_mdm.npy", test_data)

if __name__ == "__main__":
    # filter()
    # reformat()
    split()


