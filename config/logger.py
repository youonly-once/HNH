# @Time    : 2023/11/7
# @Author  : SXS
# @Github  : https://github.com/SXS-PRIVATE/HNH
import logging
import time
import os.path as osp
def get_logger():
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    log_name = now + '_log.txt'
    log_dir = './log'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)
    return logger
