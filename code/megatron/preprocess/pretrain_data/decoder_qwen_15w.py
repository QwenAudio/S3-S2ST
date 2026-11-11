import argparse
import jsonlines
import numpy as np
from encoder_qwen_15w import Encoder, ENDOFTEXT


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path for input bin files')
    parser.add_argument('--out-path', type=str, help='path for input bin files')
    parser.add_argument('--size', type=int, default=4, help='size for a token id in bytes')
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Interval between progress updates",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=256,
        help="chunk for each read",
    )

    args = parser.parse_args()

    encoder = Encoder()

    if args.size == 4:
        t = np.int32
    elif args.size == 2:
        t = np.uint16
    else:
        raise ValueError(f'unknown id size {args.size}')
    prefix = ""
    if args.out_path:
        fout = jsonlines.open(args.out_path,'w') 
    else:
        fout = None
    with open(args.path, 'rb') as fin:
        count = 0
        while True:
            data = fin.read(args.chunk * args.size)
            if not data:
                break
            data = np.frombuffer(data, dtype=t)
            data = data.astype(np.int32)
            data = data.tolist()
            if fout:
                decoded_text = prefix + encoder.decode(data)
                decoded_text_split = decoded_text.split(ENDOFTEXT)
                prefix = decoded_text_split[-1]
                for item in decoded_text_split[:-1]:
                    fout.write({"text":item})
            else: 
                print(encoder.debug(data))
                print('\n\n=========\n\n')
                if count >=5:
                    break
                count += 1

if __name__ == '__main__':
    main()

