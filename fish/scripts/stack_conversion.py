from glob import glob
import os
import volTools as volt
import fileTools as ftools

from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("image_conversion")
sc = SparkContext(conf=conf)

to_convert = [
    "/nobackup/ahrens/davis/data/raw/20160608/6dpf_cy171xcy221_f1_omr_1_20160608_170933/",
    "/nobackup/ahrens/davis/data/raw/20160608/6dpf_cy171xcy221_f2_omr_1_20160608_190404/",
    "/nobackup/ahrens/davis/data/raw/20160608/6dpf_cy171xcy221_f2_omr_1_20160608_190404/",
    "/nobackup/ahrens/davis/data/raw/20160614/5dpf_cy171xcy221_f1_caudal_omr_1_20160614_183344/",
    "/nobackup/ahrens/davis/data/raw/20160614/5dpf_cy171xcy221_f1_caudal_omr_2_20160614_185018/",
]


def image_conversion(raw_dir, source_format="stack", dest_format="klb"):
    """
    Find all files in a directory with a specified format, parallelize this list over a cluster using spark, and convert each file to a new format.

    raw_dir : string
        Directory containing files to be converted

    source_format : string, default is 'stack'
        The input format of the files to be converted. Supported formats are 'stack' and 'tif'.

    dest_format : string, default is 'klb'
        The output format of the converted files. Supported formats are 'klb' and 'hdf5'
    """
    from glob import glob

    # Data files start with `TM`
    source_glob = "{0}TM*.{1}".format(raw_dir, source_format)
    dest_glob = "{0}TM*.{1}".format(raw_dir, dest_format)

    print("Source directory: {0}".format(raw_dir))

    fnames = glob(source_glob)
    fname_rdd = sc.parallelize(fnames, numSlices=256)

    old_source = fnames
    old_dest = glob(dest_glob)

    print("pre-conversion: number of {0} files: {1}".format(dest_format, len(old_dest)))
    print(
        "pre-conversion: number of {0} files: {1}".format(
            source_format, len(old_source)
        )
    )

    convert_fun = lambda f: ftools.image_conversion(f, dest_format, wipe=True)

    if len(old_source) == 0:
        print("No {0} files found!".format(source_format))
    else:
        fname_rdd.foreach(convert_fun)

    new_dest = glob(dest_glob)
    new_source = glob(source_glob)

    print(
        "post-conversion: number of {0} files: {1}".format(dest_format, len(new_dest))
    )
    print(
        "post-conversion: number of {0} files: {1}".format(
            source_format, len(new_source)
        )
    )


for r in to_convert:
    try:
        image_conversion(r)
    except:
        print("Something went wrong processing {0}".format(r))
