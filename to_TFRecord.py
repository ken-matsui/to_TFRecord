# coding: utf-8

# Inside modules
import os
# Outside modules
import tensorflow as tf
import numpy as np
import click
from PIL import Image # pip install Pillow

@click.command()
@click.argument('filedir')
def main(filedir):
	dataset_path = "./dataset.tfrecords"
	width, height = 96, 96
	channels = 3
	datas = os.listdir(filedir)
	writer = tf.python_io.TFRecordWriter(dataset_path)
	for img_name in datas:
		# Resize
		img_obj = Image.open(filedir + "/" + img_name).convert("RGB").resize((width, height))
		# Convert to bytes
		img = np.array(img_obj).tostring()

		record = tf.train.Example(features=tf.train.Features(feature={
				"image": tf.train.Feature(
						bytes_list=tf.train.BytesList(value=[img])),
				"height": tf.train.Feature(
						int64_list=tf.train.Int64List(value=[height])),
				"width": tf.train.Feature(
						int64_list=tf.train.Int64List(value=[width])),
				"depth": tf.train.Feature(
						int64_list=tf.train.Int64List(value=[channels])),
		}))
		writer.write(record.SerializeToString())
	writer.close()

if __name__ == '__main__':
	main()