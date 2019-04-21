import csv

class Metrics:

    def __init__(self, directory):
        self.directory = directory
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
    
    def _save(self, filename, values):
        with open('{}/{}'.format(self.directory, filename), 'w') as f:
            writer = csv.writer(f)
            for value in values:
                writer.writerow([value])

    def save(self):
       self._save("train_loss.csv", self.train_loss)
       self._save("val_loss.csv", self.val_loss)
       self._save("test_loss.csv", self.test_loss)
