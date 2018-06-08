import os
from PIL import Image

def resize_jpg2png(in_path, out_path):
    # 输出所有文件和文件夹
    for file in os.listdir(in_path):

        im = Image.open(os.path.join(in_path, file))
        out = im.resize((299, 299))
        file_name = file[:-4]
        file_name_png = file_name + '.png'
        out.save(os.path.join(out_path, file_name_png))
#
# def load_images(input_dir):
#   """Read png images from input directory in batches.
#
#   Args:
#     input_dir: input directory
#     batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
#
#   Yields:
#     filenames: list file names without path of each image
#       Lenght of this list could be less than batch_size, in this case only
#       first few images of the result are elements of the minibatch.
#     images: array with all images from this batch
#   """
#   for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.jpg')):
#
#     with tf.gfile.Open(filepath, 'rb') as f:
#       image = imread(f, mode='RGB').astype(np.float)
#     # Images for inception classifier are normalized to be in [-1, 1] interval.
#     image_path = os.path.basename(filepath)
#     return  image_path, image
def main():
    in_path = 'D:\\work\AdversariaML\\NonTargetedAdversarialAttacksmaster\\adv_samples_imagenet'
    out_path = 'D:\\work\AdversariaML\\NonTargetedAdversarialAttacksmaster\\adv_samples'
    resize_jpg2png(in_path, out_path)

if __name__ == '__main__':
    main()

