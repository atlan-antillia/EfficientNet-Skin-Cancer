import os
import sys
import glob
import traceback

class DatasetStatistics:

  def __init__(self):
    pass


  def show(self, dataset_dir):
     subdirs = os.listdir(dataset_fir)
     subdirs = sorted(subdirs)
     for dir in subdirs:
       count = glob.glob(dir + "/*.jpg")
       dir = os.path.basename(dir)
       print("dir {}  count {}".format(dir, count))


if __name__ == "__main__":
  try:
    dataset_dir = ""
    if len(sys.argv) == 2:
      dataset_dir = sys.argv[1]
    else:
      raise Exception("Invalid argment")

    stat = DatasetStatistics()
    stat.show(dataset_dir)

  except:
    traceback.print_exc()