import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    # ==================================================================================================================
    # MODEL PARAMETERS
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size')
    parser.add_argument('--coded_size', type=int, default=4,
                        help='number of bits representing the encoded patch')
    parser.add_argument('--patch_size', type=int, default=8,
                        help='size for the encoded subdivision of the input image')

    # ==================================================================================================================
    # OPTIMIZATION
    # ------------------------------------------------------------------------------------------------------------------
    parser.add_argument('--')