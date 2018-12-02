# coding: utf-8

# Inside modules
import os
import sys
import threading
# Outside modules
import tensorflow as tf
import numpy as np
from PIL import Image # pip install pillow


def main(input_path, output_path):
    width, height = 96, 96
    channels = 3
    datas = os.listdir(input_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    print("Convert to TFRecord...")
    for i, img_name in enumerate(datas):
        # Resize
        img_obj = Image.open(input_path + "/" + img_name).convert("RGB").resize((width, height))
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
        print_bar(i+1, len(datas))
    writer.close()
    print("\ndone.")

def print_bar(now_c, max_c):
    def run():
        max_bar = 50
        percent = (now_c*100) // max_c
        bar_num = percent // 2
        if now_c >= max_c:
            bar = "=" * (max_bar-1) + ">"
            percent = 100
            now_c = max_c
        elif (bar_num - 1) > 0:
            bar = "=" * (bar_num-1) + ">" + " " * (max_bar-bar_num)
        elif bar_num == 1:
            bar = ">" + " " * (max_bar-1)
        else:
            bar = " " * max_bar
        sys.stdout.write("{}/{} [{}] - {}%\r".format(now_c, max_c, bar, percent))
    thread = threading.Thread(target=run)
    thread.start()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(':Usage: python %s input_path output_path' %sys.argv[0])
    else:
        main(sys.argv[1], sys.argv[2])
