import subprocess as sub
import config as cfg
import os

def correrTp(input, iterations = cfg.iterations, tolerance = cfg.tolerance):
    ejecutable = str(os.getcwd()) + "/tp2"
    cmd = [ejecutable, input, str(iterations), str(tolerance)]
    sub.run(cmd)

