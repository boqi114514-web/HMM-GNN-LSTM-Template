"""Offline entry point. No imports with training side effects before CLI parsing."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='HMM + GAT + LSTM reusable monthly research template')
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='validate local input schemas and history; no training')
    check.add_argument('--data-dir', type=Path, required=True)
    check.add_argument('--config', type=Path)
    train = commands.add_parser('train', help='train the base architecture in a NEW output directory')
    train.add_argument('--data-dir', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--config', type=Path)
    demo = commands.add_parser('demo', help='generate synthetic inputs and run all three networks on CPU')
    demo.add_argument('--out', type=Path, required=True)
    demo.add_argument('--data-dir', type=Path, help='NEW synthetic data directory; default is OUT.parent/OUT.name_data')
    commands.add_parser('test', help='run offline unit and architecture tests')
    args = parser.parse_args()
    if args.command == 'test':
        return subprocess.call([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT/'tests'), '-v'], cwd=ROOT)
    sys.path.insert(0, str(ROOT/'src'))
    if args.command in ('train', 'demo'):
        args.out = args.out.resolve()
        if args.out.exists():
            parser.error('Output already exists; choose a new directory to preserve previous results')
        os.environ['HMM_TEMPLATE_OUTPUT_DIR'] = str(args.out)
    if args.command == 'demo':
        args.data_dir = args.data_dir or args.out.with_name(args.out.name + '_data')
        os.environ['HMM_TEMPLATE_PROFILE'] = 'smoke'
        # Configure paths before the generator or any model imports config.
        os.environ['HMM_TEMPLATE_DATA_DIR'] = str(args.data_dir.resolve())
        from demo_data import generate_demo
        generate_demo(args.data_dir.resolve())
    else:
        os.environ['HMM_TEMPLATE_PROFILE'] = 'reference'
        os.environ['HMM_TEMPLATE_DATA_DIR'] = str(args.data_dir.resolve())
    if getattr(args, 'config', None):
        os.environ['HMM_TEMPLATE_CONFIG'] = str(args.config.resolve())
    import config
    summary = config.check_data()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.command == 'check':
        return 0
    import torch
    torch.set_num_threads(1)
    from run_original_architecture import main as train_all
    train_all(output_dir=args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
