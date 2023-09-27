from base import *

class LogisticRegression(torch.nn.Module):
    """
    Our LogisticRegression class as a linear regression with a sigmoid function
    applied to the output.

    Example:
        model = LogisticRegression(input_dim, output_dim) # Initialize
        y_pred_proba_train = model(X_train) # Predict probabilities
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, dtype=float_pt)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs