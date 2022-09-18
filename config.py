import argparse


def load_args():
    parser = argparse.ArgumentParser()

    # Pre training
    parser.add_argument('--base_dir', type=str, default='.\\data')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--checkpoints', type=str, default='.\\checkpoints')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--M', type=int, default=7)
    parser.add_argument('--intervals', type=int, default=1)

    # Network
    parser.add_argument('--proj_hidden', type=int, default=4096)
    parser.add_argument('--proj_in', type=int, default=2048)
    parser.add_argument('--proj_out', type=int, default=2048)

    # Down Stream Task
    # parser.add_argument('--down_lr', type=float, default=0.03)
    # parser.add_argument('--down_epochs', type=int, default=200)
    # parser.add_argument('--down_batch_size', type=int, default=256)


    # distributed train
    parser.add_argument('--syncBN', type=bool, default=True)
     不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args
