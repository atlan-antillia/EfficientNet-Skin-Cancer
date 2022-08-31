import os
import sys
import glob
from tkinter.tix import MAX
import traceback
from matplotlib import pyplot as plt

class DatasetStatistics:

  def __init__(self, fig_tilte_font_size=18, title_fontsize=14, label_fontsize=12):
    self.fig_title_fontsize = fig_tilte_font_size
    self.title_fontsize = title_fontsize
    self.label_fontsize = label_fontsize


  def setvalue(self, graph, height):
    for rect in graph:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom')

  def getCountMax(self, subdirs):
    MAX = 0
    for dir in subdirs:
       # ./dataset_dir/*.test , 
       dir = os.path.join(dataset_dir, dir) 
       labels = os.listdir(dir)
       for label in labels:
         label_dir = os.path.join(dir, label)
         files = glob.glob(label_dir + "/*.jpg")
         count = len(files)
         if count>=MAX:
          MAX = count
    return MAX

  def getCount(self, dir, labels):
    counts = []
    for label in labels:
      label_dir = os.path.join(dir, label)
      files = glob.glob(label_dir + "/*.jpg")
      count = len(files)
      counts.append(count)
    print("--- labels {} couns {}".format(labels, counts))
    return counts

  def plot(self, dataset_dir):
    title   = os.path.basename(dataset_dir)
    subdirs = os.listdir(dataset_dir)
    # sub_dirs = test, train 
    #plt.title(dataset_dir)

    fig = plt.figure(figsize=(12,6))
    fig.suptitle(dataset_dir, fontsize=self.fig_title_fontsize, weight='bold')
    plt.subplots_adjust(wspace=0.5,)
    subdirs = sorted(subdirs)
    subdirs.reverse()
    i = 0
    YMAX = self.getCountMax(subdirs)
    YMAX = YMAX + int(YMAX/10)

    for dir in subdirs:
       # ./dataset_dir/*.test , 
       dir = os.path.join(dataset_dir, dir) 
       labels = os.listdir(dir)
       print("--- labels {}".format(labels))
       #[label1, label2]
       counts = self.getCount(dir, labels)

       ax = fig.add_subplot(1, 2, 1+i)
       ax.set_ylim(0, YMAX)

       i += 1
       graph = plt.bar(labels, counts)
       self.setvalue(graph, counts)
       #plt.xlabel("Labels", fontsize=self.label_fontsize)
       ax.set_xlabel("Labels", fontsize = self.label_fontsize, weight = 'bold')
       ax.set_ylabel("Count", fontsize = self.label_fontsize, weight = 'bold')
       title = os.path.basename(dir)
       ax.set_title(title, fontsize = self.title_fontsize, weight = 'bold')


    #plt.show()
    filename = os.path.basename(dataset_dir)
    filename = dataset_dir.replace("/", "_")
    filename = filename.replace("\\", "_")
    filename = filename.replace(".", "")
    figfilename = os.path.join("./", filename + ".png")
    fig.savefig(figfilename)


# python DatasetStatistics.py ./Resampled_HAM10000
if __name__ == "__main__":
  dataset_dir = "./Resampled_HAM10000/"
  try:
    #dataset_dir = ""
    if len(sys.argv) == 2:
      dataset_dir = sys.argv[1]

    stat = DatasetStatistics()
    stat.plot(dataset_dir)

  except:
    traceback.print_exc()