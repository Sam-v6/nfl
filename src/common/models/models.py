import torch



class MLP(torch.nn.Module):
    def__init__(self, input_size, hidden_size, output_size):
        super().__init__()


        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

        self.forward(self, x):
            return self.layers(x)