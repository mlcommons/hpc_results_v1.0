import os
import shutil

def stage_files(input_dir, output_dir, stage_rank, stage_size):
    """
    Simple staging function which copies all files from input_dir to output_dir
    by stage_size participants.
    """
    os.makedirs(output_dir, exist_ok=True)
    filenames = os.listdir(input_dir)

    for filename in filenames[stage_rank::stage_size]:
        shutil.copy(os.path.join(input_dir, filename),
                    os.path.join(output_dir, filename))
