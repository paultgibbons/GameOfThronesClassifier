import glob
import os

filelist = glob.glob("pickles/*.*")
for f in filelist:
    os.remove(f)