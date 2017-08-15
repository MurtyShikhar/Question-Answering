import zipfile
from squad_preprocess import *


if __name__ == '__main__':
    glove_base_url = "http://nlp.stanford.edu/data/"
    glove_filename = "glove.6B.zip"
    prefix = os.path.join("download", "dwr")

    print("Storing datasets in {}".format(prefix))

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    glove_zip = maybe_download(glove_base_url, glove_filename, prefix, 862182613L)
    glove_zip_ref = zipfile.ZipFile(os.path.join(prefix, glove_filename), 'r')

    glove_zip_ref.extractall(prefix)
    glove_zip_ref.close()
