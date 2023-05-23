import subprocess as sub
import config as cfg
import os

def corerTp(input, iterations = cfg.iterations, tolerance = cfg.tolerance):
    tp2 = str(os.getcwd()) + "/tp2"
    cmd = [tp2, input, str(iterations), str(tolerance)]
    sub.run(cmd)

