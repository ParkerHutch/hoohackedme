import sys


class Analysis:
    def __init__(self, model_data, train_data):
        self.train_data = train_data
        self.model_data = model_data

    def run_analysis(self):
        number_correct = 0
        for password in self.model_data:
            if password in self.train_data:
                number_correct += 1
        print("Number of Passwords Correctly Guessed:", number_correct)


def main():
    model_output = sys.argv[1]
    train_data = sys.argv[2]
    analyzer = Analysis(model_data=model_output, train_data=train_data)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
