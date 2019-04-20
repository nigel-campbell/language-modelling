import csv

class Metrics:

    def __init__(self, directory):
        self.directory = directory
        self.loss_history = []

    def save(self):
        with open('{}/{}'.format(self.directory, 'loss_history.csv'), 'w') as f:
            writer = csv.writer(f)
            for value in self.loss_history:
                writer.writerow([value])
