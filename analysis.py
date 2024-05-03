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
        
        print("List of correct ones:", passwords)
        print("Number of Passwords Correctly Guessed:", number_correct)


def main():
    model_file = sys.argv[1]
    train_file = sys.argv[2]
    output_filename = sys.argv[3]

    

    train_data_set = set()
    with open(train_file, "r", errors='ignore') as tf:
        for line in tf:
            train_data_set.add(line)

    model_output = []
    correct_guesses = 0

    
    with open(output_filename, 'w') as analysis_output_file:
        analysis_output_file.write('Guess #, Correct Guesses Total\n')
        with open(model_file, "r") as mf:
            for idx, line in enumerate(mf):
                if line in train_data_set:
                    correct_guesses += 1
                    analysis_output_file.write(f'{idx + 1}, {correct_guesses}\n')
                    # print(idx, correct_guesses)
            # model_output.append(line)
            # model_output.append(mf.readlines())
    print(f'Finished; {correct_guesses} correct guesses found in {model_file}')
    # analyzer = Analysis(model_data=model_output, train_data=train_data_set)
    # analyzer.run_analysis()


if __name__ == "__main__":
    # !!!! this commented out code below was just to create the txt file with training data (lines)
    # lines = []
    # with open('rockyou.txt', 'r', encoding='utf-8', errors='ignore') as data_file:
    #     for line in data_file:
    #         lines.append(line.rstrip())
    #     data_file.close()
    #
    # print(len(lines))
    #
    # data = lines[:1000]
    # with open("train_data.txt", "w",encoding='utf-8', errors='ignore' ) as train_data_file:
    #     for line in data:
    #         train_data_file.write(line + "\n")
    #     train_data_file.close()

    main()
