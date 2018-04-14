import imp
import os.path as osp
import logging


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)


logger_agent = logging.getLogger('Agent')
logger_agent.setLevel(logging.INFO)
fh_agent = logging.FileHandler('./agent.log')
sh = logging.StreamHandler()
fm = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > [%(name)s] %(message)s')
fh_agent.setFormatter(fm)
sh.setFormatter(fm)
logger_agent.addHandler(fh_agent)
logger_agent.addHandler(sh)
