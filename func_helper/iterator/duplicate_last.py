class DuplicateLast:
    def __init__(self, *arg):
        self.args = arg

    def __len__(self):
        return len(self.args)

    def __getitem__(self, n):
        return self.args[n] if len(self) > n else self.args[-1]
