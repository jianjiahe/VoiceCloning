import logging
from operator import mul

import tensorflow as tf
from functools import reduce
from tensorflow.contrib import graph_editor as ge


def check_op_name(regex='Block./layer.*/concat.*'):
    ops = tf.get_default_graph().get_operations()
    concat_ops_list = ge.filter_ops_from_regex(ops, regex)
    for op in ops:
        if op not in concat_ops_list and op.outputs:
            print(op)
            logging.info(' out: {0}'.format(op.outputs[0]))


def check_param_num():
    logging.info('checking parameter number...')
    num = 0
    param_dict = {}
    for var in tf.trainable_variables():
        shape = var.get_shape()
        cur_num = reduce(mul, [dim.value for dim in shape], 1)
        param_dict[var.name] = cur_num
        num += cur_num
    logging.info('total parameter: {0}'.format(num))
    logging.info(param_dict)