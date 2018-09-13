from rasa_nlu.training_data import load_data
from rasa_nlu import config
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter

import argparse

data = "data/train.md"
configs = "config.yml"
model_dir = "./models"

#def train_nlu(data, configs, model_dir):

def train_nlu():
    training_data = load_data(data)
    trainer = Trainer(config.load(configs))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir, fixed_model_name = 'default')

# run the model and test it
def run_nlu():
    interpreter = Interpreter.load('./models/default/default')
    print(interpreter.parse("Tell me about Prague"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trains nlu model")
    parser.add_argument("-t",
                        choices=["train", "run_nlu"],
                        help="train or run nlu")

    task = parser.parse_args().t
    
    if task == "train":
        train_nlu()
    elif task == "run_nlu":
        run_nlu()
    else:
        warnings.warn("Need to pass 'train' or 'run_nlu' as argument")
        exit(1)