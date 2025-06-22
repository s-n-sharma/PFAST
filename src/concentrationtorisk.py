import numpy as np

class RiskPrediction:
    def __init__(self):
        self.ds = []
        """
        PFBS = 0
        PFBA = 1
        PFHxS = 2
        PFHxA = 3
        PFOS = 4
        PFOA = 5
        PFHpA = 6
        PFPeA = 7
        PFNA = 8
        ADONA = 9
        PFPeS = 10
        FTSA = 11
        """
        self.ds.append((np.array([3.7, 0, 16, 5.9, 28, 5, 2.7, 0, 0, 0, 0, 0, 61]), "M"))
        self.ds.append((np.array([13, 6.3, 79, 20, 120, 9.2, 13, 4.7, 5.7, 2.1, 6.3, 0, 259]), "H"))
        self.ds.append((np.array([0, 5.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.2]), "N"))
        self.ds.append((np.array([3.2, 4.5, 4.1, 2.4, 6.4, 2.5, 0, 2.2, 0, 0, 0, 0, 20]), "L"))
        self.ds.append((np.array([0, 0, 0, 0, 1.2, 0, 0, 0, 0, 0, 0, 0, 1.2]), "N"))
        self.ds.append((np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), "N"))
        self.ds.append((np.array([4.2, 3.2, 8.8, 2.8, 9.9, 3.5, 1.1, 2, 5.2, 0, 0, 0, 29]), "L"))
        self.ds.append((np.array([3.7, 0, 17, 7, 18, 12, 4.6, 2.8, 0, 0, 0, 0, 52]), "M"))
        self.ds.append((np.array([6.6, 2.4, 15, 4.9, 16, 13, 5.2, 5.8, 0, 0, 0, 2.6, 46]), "M"))
    
    def addExample(self, vec, label):
        self.ds.append((vec/np.linalg.norm(vec), label))
    
    def addCategory(self, category):
        self.categories.append(category)
    
    def predict(self, vec):
        norm_vec = vec + 0.0001*np.ones(len(vec))
        if np.linalg.norm(norm_vec) < 0.2:
            return "N"
        if not len(self.ds):
            return "n/a"
        norm_vec = norm_vec/np.linalg.norm(norm_vec)
        max_sim = np.dot(norm_vec, self.ds[0][0])
        ret = self.ds[0][1]
        for ex in self.ds:
            sim = np.dot(norm_vec, ex[0])
            if sim > max_sim: sim, ret = max_sim, ex[1]
        return ret






