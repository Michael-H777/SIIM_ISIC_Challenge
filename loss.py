from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Accuracy:
    
    def __call__(self, prediction, target):
        prediction = [round(item) for item in prediction]
        return accuracy_score(target, prediction) 
    
class F1:
    
    def __call__(self, prediction, target):
        prediction = [round(item) for item in prediction]
        return f1_score(target, prediction) 
    
class Roc_Auc:
    
    def __call__(self, prediction, target): 
        return roc_auc_score(target, prediction) 
    
