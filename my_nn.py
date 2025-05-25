import torch.nn as nn

nn.Module.dump_patches = True

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = 2  # 0 - ничего не делать, 1 - прыгнуть
        
        # Параметры обучения
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.12
        self.number_of_iterations = 200000
        self.replay_memory_size = 70000
        self.minibatch_size = 32

       # self.input_norm = nn.BatchNorm1d(5*6) 

        self.fc_input = nn.Linear(4*6, 128)  # 4 входных параметров
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        
        self.output = nn.Linear(64, self.number_of_actions)

    def forward(self, x):
        # x должен быть тензором формы [batch_size, 3]
        # где 3 числа: [y-координата птички, y-координата нижней трубы, y-координата верхней трубы]
        
        out = self.fc_input(x)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.relu4(out)
        
        out = self.output(out)
        
        return out