import os


if __name__ == "__main__":
    root_dir = "/hkfs/work/workspace/scratch/qv2382-mlperf-cosmo-data/new_npy_files"

    for tv in ["train", "validation"]:
        # self.label_files = sorted(glob.glob(os.path.join(self.root_dir, self.prefix_label)))

        target_dir = os.path.join(root_dir, tv)  # "validation"
        target_files = os.listdir(target_dir)
        print(tv, target_dir)

        data_files = sorted(list(filter(lambda x: x.endswith("data.npy"), target_files)))
        label_files = sorted(list(filter(lambda x: x.endswith("label.npy"), target_files)))
        # two write: files_data.lst, files_label.lst
        out_dir = f"/hkfs/work/workspace/scratch/qv2382-mlperf-cosmo-data/h5_files/{tv}"
        with open(os.path.join(out_dir, "files_data.lst"), "a") as fp:
            for df in data_files:
                fp.write('%s\n' % df)
        with open(os.path.join(out_dir, "files_label.lst"), "a") as fp:
            for df in label_files:
                fp.write('%s\n' % df)
