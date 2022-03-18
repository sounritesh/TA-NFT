import torch
import pytz

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UTC = pytz.utc