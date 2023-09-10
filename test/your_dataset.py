from torch.utils.data import Dataset

class YourDataset(Dataset):
    def __init__(self, S, A, R, Sp, done, eligibility_matrix):
        self.S = S
        self.A = A
        self.R = R
        self.Sp = Sp
        self.done = done
        self.eligibility_matrix = eligibility_matrix

    def __len__(self):
        return len(self.S)

    def __getitem__(self, idx):
        return self.S[idx], self.A[idx], self.R[idx], self.Sp[idx],  self.done[idx], self.eligibility_matrix[idx]
    