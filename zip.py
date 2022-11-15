import tarfile
import os


def tar_folder(output_filename: str, source_dir: str):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


OUT_FILE = 'tf-models.tar.gz'

SOURCE_FILE = "tf-models"

tar_folder(output_filename=OUT_FILE, source_dir=SOURCE_FILE)
