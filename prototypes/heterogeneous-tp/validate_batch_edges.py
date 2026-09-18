"""Exercise EOS accounting and rejection of corruption after the initial gate."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    source = (source_dir / 'batch_tp.py').read_text()
    # Mutants are diagnostic fixtures, never used for performance results.
    corrupt = source.replace('            def generate(n_new, stamps=None):',
        '            generation_calls = 0\n            def generate(n_new, stamps=None):\n'
        '                nonlocal generation_calls\n                generation_calls += 1')
    corrupt = corrupt.replace('                    nxt = choice.to(torch.int64).tolist()',
        '                    nxt = choice.to(torch.int64).tolist()\n'
        '                    if generation_calls > args.warmups + 1 and i == 0:\n'
        '                        nxt[0] = (nxt[0] + 1) % vocab')
    eos = source.replace('            for pid in prompts:', '            for seq_index, pid in enumerate(prompts):')
    eos = eos.replace('                    seq.append(nxt)',
        '                    if seq_index == 0 and len(seq) == 1:\n                        nxt = EOS\n'
        '                    seq.append(nxt)')
    eos = eos.replace('                    nxt = choice.to(torch.int64).tolist()',
        '                    nxt = choice.to(torch.int64).tolist()\n'
        '                    if i == 1:\n                        nxt[0] = EOS')
    summary = []
    with tempfile.TemporaryDirectory(prefix='batch-edges-') as temporary:
        for name, text in [('timed-corruption', corrupt), ('early-eos', eos), ('one-token', source)]:
            script = Path(temporary) / (name + '.py')
            script.write_text(text)
            output = args.out / name
            output.mkdir(exist_ok=True)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            processes, files = [], []
            count = 1 if name == 'one-token' else 8
            cmd = [sys.executable, str(script), '--mode', 'tp', '--batch', '4',
                   '--device', 'cpu', '--prompt-tokens', '12', '--new-tokens', str(count),
                   '--warmups', '1', '--runs', '2']
            try:
                for rank in range(2):
                    stdout = (output / f'rank{rank}.json').open('w')
                    stderr = (output / f'rank{rank}.log').open('w')
                    files.extend([stdout, stderr])
                    env = dict(os.environ, PYTHONPATH=str(source_dir), RANK=str(rank),
                               WORLD_SIZE='2', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
                    processes.append(subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr))
                codes = [p.wait(timeout=180) for p in processes]
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                for f in files:
                    f.close()
            ranks = json.loads((output / 'rank0.json').read_text())['ranks']
            if name == 'timed-corruption':
                assert codes == [1, 1]
                assert all(r['all_sequences_match_reference'] for r in ranks)
                assert all(not r['valid'] and not r['runs'] and len(r['invalid_runs']) == 2 for r in ranks)
            else:
                assert codes == [0, 0]
                assert all(r['valid'] for r in ranks)
                for rank in ranks:
                    for run in rank['runs']:
                        lengths = [len(ids) for ids in run['produced_ids']]
                        assert lengths == ([2, 8, 8, 8] if name == 'early-eos' else [1]*4)
                        assert run['generated_tokens_total'] == sum(lengths)
                        assert run['decode_tokens_total'] == sum(n-1 for n in lengths)
                        assert run['reduction_counters']['collectives'] == 24*max(lengths)
                        if name == 'one-token':
                            assert run['aggregate_decode_tokens_per_s'] is None
            summary.append({'case': name, 'exit_codes': codes, 'passed': True})
            print(json.dumps(summary[-1]), flush=True)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
