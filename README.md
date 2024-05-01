# hoohackedme
CS 6501 (NetSec) project

## Requirements
- Linux

## Setup
Run [setup.sh](setup.sh) to download the `rockyou.txt` password list.

## To Run the RNN script:

We are saving the progress of the model using Pickle. The necessary variables are saved in "saved.pickle". We save the variables when the smooth loss reaches below 10%, because that seemed like a good enough metric for us to us. 

To train, simply run: `python3 rnn3.py "train"`. We could add something to change the amount of loss there, to see how much better or worse the passwords are with those poasswords. 

To produce passwords, simply run `python3 rnn3.py "run" <number>` where `number` is the number of passwords you would like the model to produce. 

When you train the model, it automatically saves the necessary variables in pickle. When you run the model, it will load those variables into memory and use them to produce passwords. Kind of fire. 