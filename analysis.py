import sys


class Analysis:
    def __init__(self, model_data, train_data):
        self.train_data = train_data
        self.model_data = model_data

    def run_analysis(self):
        number_correct = 0
        passwords = []
        for password in self.model_data:
            if password in self.train_data:
                number_correct += 1
                passwords.append(password)
        print("Number of Passwords Correctly Guessed:", number_correct)
        print("List of correct ones:", passwords)


def main():
    model_file = sys.argv[1]
    train_file = sys.argv[2]

    model_output = []
    with open(model_file, "r") as mf:
        model_output.append(mf.readlines())

    train_data = []
    with open(train_file, "r") as tf:
        train_data.append(tf.readlines())

    analyzer = Analysis(model_data=model_output, train_data=train_data)
    analyzer.run_analysis()


if __name__ == "__main__":
    # !!!! this commented out code below was just to create the txt file with training data (lines)
    lines = []
    with open('rockyou.txt', 'r', encoding='utf-8', errors='ignore') as data_file:
        for line in data_file:
            lines.append(line.rstrip())
        data_file.close()

    print(len(lines))

    data = lines[:1000]
    with open("train_data.txt", "w",encoding='utf-8', errors='ignore' ) as train_data_file:
        for line in data:
            train_data_file.write(line + "\n")
        train_data_file.close()

    main()
