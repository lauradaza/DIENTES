import argparse

parser = argparse.ArgumentParser()
## for distributed training
parser.add_argument(
    "--dist",
    dest="dist",
    type=bool,
    default=False,
    help="distributed training or not",
)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--device")
## logging
parser.add_argument(
    "--log_name", default="inference", help="The path resume from checkpoint"
)
## model load
parser.add_argument(
    "--model_path",
    default="./src/pretrained_weights/final_model.pth",
    help="The path of pretrain model",
)
parser.add_argument("--num_class", default=48, type=int, help="Number of classes")
parser.add_argument(
    "--entities",
    action="store_true",
    default=False,
    help="Binary segmentation of entities",
)
parser.add_argument(
    "--word_embedding",
    default="./src/pretrained_weights/txt_encoding.pth",
    help="The path of word embedding",
)

## dataset
parser.add_argument(
    "--datasetkey",
    nargs="+",
    default=["1"],
    help="datasets to use in this run",
)
parser.add_argument("--init_test", default=0, type=int, help="batch size")
parser.add_argument("--end_test", default=0, type=int, help="batch size")

parser.add_argument(
    "--data_root_path",
    default="/path/to/data/",
    help="data root path",
)
parser.add_argument(
    "--data_txt_path", default="./dataset/dataset_list/", help="data txt path"
)
parser.add_argument("--batch_size", default=3, type=int, help="batch size")
parser.add_argument(
    "--num_workers", default=5, type=int, help="workers numebr for DataLoader"
)
parser.add_argument(
    "--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged"
)
parser.add_argument(
    "--a_max", default=4500, type=float, help="a_max in ScaleIntensityRanged"
)
parser.add_argument(
    "--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged"
)
parser.add_argument(
    "--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged"
)
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

parser.add_argument("--phase", default="val", help="train or val or test")
parser.add_argument(
    "--store_result",
    action="store_true",
    default=False,
    help="save prediction result",
)
parser.add_argument(
    "--simple_merge",
    action="store_true",
    default=False,
    help="do not postprocess the predictions before merging them",
)
parser.add_argument("--threshold", default=0.8, type=float)

args = parser.parse_args()
