import os
from time import sleep
import sys
argv = sys.argv

file = argv[1]
if len(argv)> 2:
  sec = 60 * int(argv[2])
  print(f'wait {sec} sec')
else:
  sec = 0

sleep(sec)
# os.system(f'nohup python -u {file} 1>&2 > LOG/log_{file}.txt &')
os.system(f'python -u {file} | tee LOG/log_{file}.txt')