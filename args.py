import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser():
    parser = argparse.ArgumentParser()

    # --- Data params ---
    parser.add_argument("--dataset", type=str.upper, default="SMAP")
    # parser.add_argument("--group", type=str, default="1-1", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=100, help="Squence length")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False)

    # --- Model params ---
    # 1D conv layer
    parser.add_argument("--conv_kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.2, help="Gat hyperParameter")

    # TCN
    parser.add_argument("--tcn_kernel_size", type=int, default=3)
    parser.add_argument('--tcn_levels', type=int, default=3,
                        help='# of levels (default: 8)')
    parser.add_argument('--tcn_nhid', type=int, default=32,
                        help='number of hidden units per layer (default: 150)')
    # # GRU layer
    # parser.add_argument("--gru_n_layers", type=int, default=1)
    # parser.add_argument("--gru_hid_dim", type=int, default=150)
    # # Forecasting Model
    # parser.add_argument("--fc_n_layers", type=int, default=3)
    # parser.add_argument("--fc_hid_dim", type=int, default=150)
    # # Reconstruction Model
    # parser.add_argument("--recon_n_layers", type=int, default=1)
    # parser.add_argument("--recon_hid_dim", type=int, default=150)
    # # Other


    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)
    #
    # # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)
    #
    # --- Other ---
    parser.add_argument("--comment", type=str, default="")

    parser.add_argument('--seed', type=int, default=8888,
                        help='random seed (default: 8888)')
    return parser
