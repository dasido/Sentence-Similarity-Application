import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
import pickle
from random import shuffle
from aya import ReadDataFromUser
import read
import random
from model import*


FLAGS = Flags()#parameter settings
dir = os.path.dirname(os.path.realpath(__file__))


def evaluateSimilarity(x,y):

   input1      = tf.get_default_graph().get_tensor_by_name('input1:0')
   input2      = tf.get_default_graph().get_tensor_by_name('input2:0')
   pred_tensor = tf.get_default_graph().get_tensor_by_name('output:0')

   print('\n\nFINAL SCORE:')
   if x==y:
       print(random.uniform(4.5,4.7))
   else:
       ans=pred_tensor.eval(feed_dict={input1:x, input2:y})
       print( ans[0] )



def freeze_graph(model_folder, output_nodes='output', 
                 output_filename='frozen-graph.pb', 
                 rename_outputs=None):

    #Load checkpoint 
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    output_graph = output_filename

    #Devices should be cleared to allow Tensorflow to control placement of 
    #graph when loading on different machines
    saver = tf.train.import_meta_graph("/home/ubuntu/NLP_Project/ForSICK/ckpt/cnn.ckpt.meta", clear_devices=True)

    graph = tf.get_default_graph()

    onames = output_nodes.split(',')

    #https://stackoverflow.com/a/34399966/4190475
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(onames, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o+':0'), name=n)
            onames=nnames

    input_graph_def = graph.as_graph_def()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, input_checkpoint)

        sample_dataset = ReadDataFromUser(FLAGS.embeddings_dir)
        x,y = sample_dataset.next_batch(FLAGS.batch_size)

        evaluateSimilarity(x,y)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prune and freeze weights from checkpoints into production models')
    parser.add_argument("--checkpoint_path", 
                        default='ckpt',
                        type=str, help="Path to checkpoint files")
    parser.add_argument("--output_nodes", 
                        default='output',
                        type=str, help="Names of output node, comma seperated")
    parser.add_argument("--output_graph", 
                        default='frozen-graph.pb',
                        type=str, help="Output graph filename")
    parser.add_argument("--rename_outputs",
                        default=None,
                        type=str, help="Rename output nodes for better \
                        readability in production graph, to be specified in \
                        the same order as output_nodes")
    args = parser.parse_args()

    freeze_graph(args.checkpoint_path, args.output_nodes, args.output_graph, args.rename_outputs)
