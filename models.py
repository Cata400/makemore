import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from typing import List


class BigramsProb:
    def __init__(self, data: List[str], seed: int = 42):
        self.data = data
        
        chars = sorted(list(set("".join(data))))
        
        self.stoi = {s:i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        
        self.itos = {i: s for s, i in self.stoi.items()}
        
        self.generator = torch.Generator().manual_seed(seed)
        
        self.P = None
        
    
    def fit(self, plot: bool = False):
        N = torch.zeros((27, 27), dtype=torch.int32)
        
        for w in self.data:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                idx1, idx2 = self.stoi[ch1], self.stoi[ch2]
                N[idx1, idx2] += 1
        
        if plot:
            plt.figure(figsize=(16, 16))
            plt.imshow(N, cmap='Blues')

            for i in range(27):
                for j in range(27):
                    chstr = self.itos[i] + self.itos[j]
                    plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
                    plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
            plt.axis('off')
            plt.tight_layout()

            plt.show()
            
        P = (N.float() + 1) / (N + 1).float().sum(dim=1, keepdim=True)
        setattr(self, 'P', P)
    
    
    def generate(self, num_it : int = 5) -> List[str]:
        names = []
        
        if self.P is None:
            raise ValueError("The model was not fitted! Call <<your_model>>.fit() first")
                
        for _ in range(num_it):
            idx = 0
            new_name = ""
            while True:
                p = self.P[idx]
                                
                idx = torch.multinomial(p, num_samples=1, replacement=True, generator=self.generator).item()
                
                if idx == 0:
                    names.append(new_name)
                    break
                
                new_name += self.itos[idx]
                
        return names
    
    
    def evaluate(self) -> float:
        neg_log_likelihood = 0
        n = 0
        
        if self.P is None:
            raise ValueError("The model was not fitted! Call <<your_model>>.fit() first")

        for w in self.data:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                idx1, idx2 = self.stoi[ch1], self.stoi[ch2]
                prob = self.P[idx1, idx2]
                logprob = torch.log(prob)
                neg_log_likelihood -= logprob
                n += 1
                
        return neg_log_likelihood.item() / n
    
    
class BigramsWeights:
    def __init__(self, data: List[str], seed: int = 42):
        self.data = data
        
        chars = sorted(list(set("".join(data))))
        
        self.stoi = {s:i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        
        self.itos = {i: s for s, i in self.stoi.items()}
        
        self.generator = torch.Generator().manual_seed(seed)
        self.initial_state = self.generator.get_state()
        
        self.W = torch.randn((27, 27), generator=self.generator, requires_grad=True)
        
        self.X_train, self.Y_train = None, None
        self.X_val, self.Y_val = None, None
        self.X_test, self.Y_test = None, None
        
    def _create_dataset(self):
        X, Y = [], []

        for w in self.data:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                idx1, idx2 = self.stoi[ch1], self.stoi[ch2]     
                
                X.append(idx1)
                Y.append(idx2)

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        
        X = F.one_hot(X, num_classes=27).float()
        Y = F.one_hot(Y, num_classes=27).float()
        
        indices = torch.randperm(X.shape[0])
        
        X_train, Y_train = X[indices[: int(0.8 * X.shape[0])], :], Y[indices[: int(0.8 * X.shape[0])], :] 
        X_val, Y_val = X[indices[int(0.8 * X.shape[0]): int(0.9 * X.shape[0])], :], Y[indices[int(0.8 * X.shape[0]): int(0.9 * X.shape[0])], :] 
        X_test, Y_test = X[indices[int(0.9 * X.shape[0]):], :], Y[indices[int(0.9 * X.shape[0]):], :] 
        
        setattr(self, 'X_train', X_train)
        setattr(self, 'Y_train', Y_train)
        
        setattr(self, 'X_val', X_val)
        setattr(self, 'Y_val', Y_val)
        
        setattr(self, 'X_test', X_test)
        setattr(self, 'Y_test', Y_test)
                
    
    def train_loop(self, epochs: int = 1000, lr: float = 0.01):
        self._create_dataset()
        
        self.generator.set_state(self.initial_state)

        for i in range(epochs):
            logits = self.X_train @ self.W
                                            
            train_loss = F.cross_entropy(logits, self.Y_train).mean() + 0.01 * (self.W**2).mean()
            
            self.W.grad = None
            train_loss.backward() 
            
            self.W.data -= lr * self.W.grad
            
            with torch.no_grad():
                logits = self.X_val @ self.W
                                                
                val_loss = F.cross_entropy(logits, self.Y_val).mean() + 0.01 * (self.W**2).mean()
                
            
            if i == 0 or not (i + 1) % 100:
                print(f'Epoch #{i + 1}/{epochs} Train Loss = {train_loss.item()}; Val Loss = {val_loss.item()}')
                
    
    def generate(self, num_it: int = 5) -> List[str]:
        names = []
                
        self.generator.set_state(self.initial_state)
        
        for _ in range(num_it):
            new_name = ""
            idx = 0
            
            while True:
                xenc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
                logits = xenc @ self.W
                counts = logits.exp()
                p = counts / counts.sum(1, keepdims=True)
                
                idx = torch.multinomial(p, num_samples=1, replacement=True, generator=self.generator).item()
                                
                if idx == 0:
                    names.append(new_name)
                    break

                new_name += self.itos[idx]
                
        return names
    
    
    def evaluate(self) -> float:
        if self.Y_test is None:
            raise ValueError("The model has not been trained yet! Call <<your_model>>.train_loop() first!")
        
        with torch.no_grad():
            logits = self.X_test @ self.W
                                            
            test_loss = F.cross_entropy(logits, self.Y_test).mean() + 0.01 * (self.W**2).mean()            
            
        return test_loss.item()
        
    
class TrigramsWeights:
    def __init__(self, data: List[str], seed: int = 42):
        self.data = data
        
        chars = sorted(list(set("".join(data))))
        
        self.stoi = {s:i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        
        self.itos = {i: s for s, i in self.stoi.items()}
        
        self.generator = torch.Generator().manual_seed(seed)
        self.initial_state = self.generator.get_state()
        
        self.W = torch.randn((2 * 27, 27), generator=self.generator, requires_grad=True)
        
        self.X_train, self.Y_train = None, None
        self.X_val, self.Y_val = None, None
        self.X_test, self.Y_test = None, None
        
    def _create_dataset(self):
        X, Y = [], []

        for w in self.data:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
                idx1, idx2, idx3 = self.stoi[ch1], self.stoi[ch2], self.stoi[ch3]
                
                x1 = [0.] * 27; x1[idx1] = 1.
                x2 = [0.] * 27; x2[idx2] = 1.
                y = [0.] * 27; y[idx3] = 1.
                
                X.append(x1 + x2)
                Y.append(y)

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        
        indices = torch.randperm(X.shape[0], generator=self.generator)
        
        X_train, Y_train = X[indices[: int(0.8 * X.shape[0])], :], Y[indices[: int(0.8 * X.shape[0])], :] 
        X_val, Y_val = X[indices[int(0.8 * X.shape[0]): int(0.9 * X.shape[0])], :], Y[indices[int(0.8 * X.shape[0]): int(0.9 * X.shape[0])], :] 
        X_test, Y_test = X[indices[int(0.9 * X.shape[0]):], :], Y[indices[int(0.9 * X.shape[0]):], :] 
        
        setattr(self, 'X_train', X_train)
        setattr(self, 'Y_train', Y_train)
        
        setattr(self, 'X_val', X_val)
        setattr(self, 'Y_val', Y_val)
        
        setattr(self, 'X_test', X_test)
        setattr(self, 'Y_test', Y_test)
                
    
    def train_loop(self, epochs: int = 1000, lr: float = 0.01, alpha : float = 0.01):
        self._create_dataset()
        
        self.generator.set_state(self.initial_state)

        for i in range(epochs):
            logits = self.X_train @ self.W
                                            
            train_loss = F.cross_entropy(logits, self.Y_train).mean() + alpha * (self.W**2).mean()
            
            self.W.grad = None
            train_loss.backward() 
            
            self.W.data -= lr * self.W.grad
            
            with torch.no_grad():
                logits = self.X_val @ self.W
                                                
                val_loss = F.cross_entropy(logits, self.Y_val).mean() + alpha * (self.W**2).mean()
                
            
            if i == 0 or not (i + 1) % 100:
                print(f'Epoch #{i + 1}/{epochs} Train Loss = {train_loss.item()}; Val Loss = {val_loss.item()}')
                
    
    def generate(self, num_it: int = 5) -> List[str]:
        names = []
                
        self.generator.set_state(self.initial_state)
        
        for _ in range(num_it):
            new_name = ""
            idx1 = 0
            idx2 = torch.randint(1, 27, size=(1, ), generator=self.generator).item()
            
            while True:
                xenc1 = F.one_hot(torch.tensor([idx1]), num_classes=27).float()
                xenc2 = F.one_hot(torch.tensor([idx2]), num_classes=27).float()
                xenc = torch.concat((xenc1, xenc2), dim=-1)
                logits = xenc @ self.W
                counts = logits.exp()
                p = counts / counts.sum(1, keepdims=True)
                
                idx3 = torch.multinomial(p, num_samples=1, replacement=True, generator=self.generator).item()
                                
                if idx3 == 0:
                    names.append(new_name)
                    break

                new_name += self.itos[idx3]
                
                idx1, idx2 = idx2, idx3
                
        return names
    
    
    def evaluate(self, alpha : float = 0.01) -> float:
        if self.Y_test is None:
            raise ValueError("The model has not been trained yet! Call <<your_model>>.train_loop() first!")
        
        with torch.no_grad():
            logits = self.X_test @ self.W
                                            
            test_loss = F.cross_entropy(logits, self.Y_test).mean() + alpha * (self.W**2).mean()            
            
        return test_loss.item()
            
    
if __name__ == '__main__':
    names = open('names.txt', 'r').read().splitlines()
        
    
    # ### Bigram model from probability distribution
    # model = BigramsProb(data=names)
    
    # model.fit()
    # print(f"5 new names generated by the model: {model.generate()}")
    # print(f"Negative Log Likelihood: {model.evaluate()}")
    
    # ### Bigram model from weights learning (equivalent to a single layer of a NN)
    # model = BigramsWeights(data=names)
    
    # model.train_loop()
    # print(f"5 new names generated by the model: {model.generate()}")
    # print(f"Test Cross Entropy Loss: {model.evaluate()}")
    
    ### Trigram model from weights learning (equivalent to a single layer of a NN)
    model = TrigramsWeights(data=names, seed=2147483647)
    
    model.train_loop(epochs=200, lr=50, alpha=0)
    print(f"5 new names generated by the model: {model.generate()}")
    print(f"Test Cross Entropy Loss: {model.evaluate(alpha=0)}")