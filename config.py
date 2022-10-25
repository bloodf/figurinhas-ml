"""## Configuration Class
A simple class to manage configuration
"""


class Config:
    # files location
    training_dir = "./fwc-album/"
    testing_dir = "./fwc-album/"

    # train_batch_size = 24
    train_batch_size = 24 * 2

    # train_batch_size = 8
    train_number_epochs = 300
