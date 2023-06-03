import subprocess as sub
import config as cfg
import os

def correrTp(input, k = None):
    ejecutable = str(os.getcwd()) + "/tp2"
    if k:
        cmd = [ejecutable, input, str(cfg.iterations), str(cfg.tolerance), str(k)]
    else:
        cmd = [ejecutable, input, str(cfg.iterations), str(cfg.tolerance)]
    sub.run(cmd)

