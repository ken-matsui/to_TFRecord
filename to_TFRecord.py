# coding: utf-8

# Inside modules
import os
import sys
# Outside modules
import tensorflow as tf
import numpy as np
from PIL import Image # pip install Pillow


def main(file_path):
	dataset_path = "./dataset.tfrecord"
	width, height = 96, 96
	channels = 3
	datas = os.listdir(file_path)
	writer = tf.python_io.TFRecordWriter(dataset_path)
	print("Convert to TFRecord...")
	for i, img_name in enumerate(datas):
		# Resize
		img_obj = Image.open(file_path + "/" + img_name).convert("RGB").resize((width, height))
		# Convert to bytes
		img = np.array(img_obj).tostring()

		record = tf.train.Example(features=tf.train.Features(feature={
				"image": tf.train.Feature(
						bytes_list=tf.train.BytesList(value=[img])),
				"height": tf.train.Feature(
						int64_list=tf.train.Int64List(value=[height])),
				"width": tf.train.Feature(
						int64_list=tf.train.Int64List(value=[width])),
				"channels": tf.train.Feature(
						int64_list=tf.train.Int64List(value=[channels])),
		}))
		writer.write(record.SerializeToString())

		bar, percent = calc_bar(i+1, len(datas))
		sys.stdout.write("\r{}/{} [{}] - {}%".format(i+1, len(datas), bar, percent))
	writer.close()
	print("\ndone.")

def calc_bar(now_count, max_count):
	max_bar_size = 50
	percent = (now_count*100) // max_count
	bar_num = percent // 2
	bar = ""
	if (bar_num - 1) > 0:
		for _ in range(bar_num - 1):
			bar += "="
		bar += ">"
		for _ in range(max_bar_size - bar_num):
			bar += " "
	elif bar_num == 1:
		bar = ">"
		for _ in range(max_bar_size - 1):
			bar += " "
	else:
		for _ in range(max_bar_size):
			bar += " "
	return bar, percent

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print(':Usage: python %s file_path' %sys.argv[0])
	else:
		main(sys.argv[1])