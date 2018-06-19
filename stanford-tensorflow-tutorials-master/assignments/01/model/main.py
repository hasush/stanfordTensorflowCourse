import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tdqm

from model import Model

def main():

	# Creat the graph.
	tf.logging.info('Loading graph...')
	model = Model()
	tf.logging.info('Graph loaded...')

	# Create the supervisor to monitor progress and create checkpoints.
	sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

	# Execute the graph for training or inference.
	if cfg.is_training:
		tf.logging.info('Starting training...')
		train(model, sv)
		tf.logging.info('Ending training...')
	else:
		evaluation(model, sv)


if __name__ == "__main__":

	# Run the main method.
	tf.app.run()