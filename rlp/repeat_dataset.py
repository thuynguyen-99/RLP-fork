from torch.utils.data import Dataset


class RepeatPatchDataset(Dataset):
    """
    Repeats the base dataset multiple times.
    Args:
        base_ds (Dataset): The base dataset to be repeated.
        repeat (int): Number of times to repeat the base dataset.
    """

    def __init__(self, base_ds, repeat: int = 8):
        self.base = base_ds
        self.repeat = max(1, int(repeat))

    def __len__(self):
        return len(self.base) * self.repeat

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]
