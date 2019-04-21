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
            for value in self.loss_history:
                writer.writerow([value])

    def save(self):
       _save("train_loss.csv", self.train_loss)
       _save("val_loss.csv", self.val_loss)
       _save("test_loss.csv", self.test_loss)
