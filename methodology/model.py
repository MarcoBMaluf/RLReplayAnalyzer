import torch
import torch.nn as nn

MODEL_PATH = '/Users/marcomaluf/Desktop/Unfinished Projects/paper/models/best_LSTM_state_dict.pth'

class xG_LSTM(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, output_dim=1, latent_dim=64, num_layers=3):
        super(xG_LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, 
                                  self.num_layers, bidirectional=False, batch_first=True)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.latent_dim, self.output_dim)
        )

    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
            hidden_states = (h_0, c_0)

        x, hidden_states = self.lstm_layer(x)
        x = self.fc_layer(x[:, -1, :])

        return x.squeeze(-1)
    
    def predict_xG(self, x):
        with torch.no_grad():
            logit = self.forward(x)
        xG = torch.sigmoid(logit)

        return xG

def initialize_model():
    model = xG_LSTM(hidden_dim=256, latent_dim=128)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    return model

# def predict_xG(model, x):

#     with torch.no_grad():
#         logit = model(x)
    
#     xG = torch.sigmoid(logit)

#     return xG