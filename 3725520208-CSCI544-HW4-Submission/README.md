I have commented the code for training the models, and have used the model files that I have created to generate the predictions.

Execute the commands in the following order to use the model files and generate the prediction files. (dev1.out, test1.out, dev2.out and test2.out)

Once the files are generated, then run the next set of commands to evaluate the predictions.

Commands:

python 3725520208_CSCI544_HW4.py
python eval.py -p dev1.out -g data/dev
python eval.py -p dev2.out -g data/dev
