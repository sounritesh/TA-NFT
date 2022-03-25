from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef

def classification_report(targets, outputs):
    targets = targets.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy() >= 0.5
    
    prec, recall, fscore, _ = precision_recall_fscore_support(targets, outputs)
    mcc = matthews_corrcoef(targets, outputs)
    
    return prec, recall, fscore, mcc 