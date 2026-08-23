"""Turn a benchmark run into a table and append it to this folder's README.md.

Kept as a script rather than done by hand so that entries are always shaped the
same way. The point of the file is the trend, and a trend is unreadable if every
row was formatted by a different mood.

    python3 anagnorisis_core/benchmarks/render_results.py results.json \
        --version 0.4.9 --machine "RTX 4060 8GB / 30GB RAM" [--note "..."]
"""
import argparse
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _dur(seconds):
    """Seconds, plus hours once the number stops being readable as seconds."""
    if seconds is None:
        return '—'
    if seconds >= 3600:
        return f'{seconds:,.0f}s ({seconds / 3600:.1f}h)'
    return f'{seconds:g}s'


def render(payload, version, machine, note=None, date=None):
    target = payload['target']
    n = payload['files_measured']
    date = date or datetime.date.today().isoformat()

    lines = [
        f'### {version} — {date}',
        '',
        f'- **Machine:** {machine}',
        f'- **Measured on:** {n} indexed files (demo set); '
        f'{target:,}-file figures are extrapolated from the per-file cost',
        f'- **Listing the files:** {payload["list_seconds"]}s for {n} files '
        f'({payload["list_per_file_us"]} µs/file) through the cache',
    ]
    if payload.get('startup_seconds') is not None:
        lines.append(
            f'- **Process startup:** the first `collect_files` in a fresh process '
            f'took {payload["first_call_seconds"]}s, of which '
            f'~{payload["startup_seconds"]}s is one-time imports (pyfilesystem, '
            f'the media-type taxonomy) rather than walking. A CLI invocation pays '
            f'this once; the app pays it at boot.')
    if note:
        lines.append(f'- **Note:** {note}')
    listing = payload.get('listing')
    if listing:
        lines += [
            '',
            '**Finding the files** — cached per directory by mtime, so three regimes:',
            '',
            '| Regime | Time | µs/file | '
            f'Est. @{target:,} |',
            '|---|---|---|---|',
            f'| uncached (our cache empty) | {listing["uncached_s"]}s '
            f'| {listing["uncached_per_file_us"]} | {listing[f"est_uncached_{target}_s"]}s |',
            f'| cold (cache on disk) | {listing["cold_s"]}s '
            f'| {listing["cold_per_file_us"]} | {listing[f"est_cold_{target}_s"]}s |',
            f'| warm (cache in RAM) | {listing["warm_s"]}s ±{listing["warm_stdev_s"]} '
            f'| {listing["warm_per_file_us"]} | {listing[f"est_warm_{target}_s"]}s |',
            '',
            'The *uncached* row is measured with our cache empty but the operating '
            'system\'s own directory cache likely warm, so it is a floor rather than '
            'a true first-ever scan — dropping the OS cache needs root on the host.',
        ]

    indexing = payload.get('indexing')
    if indexing:
        lines += [
            '',
            '**Building the index** — embedding each file\'s content, and embedding '
            'its description text. This does *not* include writing the description '
            'in the first place; that is the table below, and it dominates.',
            '',
            '| Media type | Files | Content s/file | Descriptions s/file |',
            '|---|---|---|---|',
        ]
        for name, row in indexing['per_type'].items():
            lines.append(
                f'| `{name}` | {row["files"]} | {row["content_per_file_s"]} '
                f'| {row["descriptions_per_file_s"]} |')
        totals = indexing['totals']
        lines += [
            '',
            f'| Phase | Total | s/file | Est. @{target:,} |',
            '|---|---|---|---|',
        ]
        for phase, t in totals.items():
            lines.append(f'| {phase} | {t["seconds"]}s | {t["per_file_s"]} '
                         f'| {_dur(t[f"est_{target}_s"])} |')
        lines += [
            f'| **both, whole library** | — | — '
            f'| **{_dur(indexing[f"est_total_{target}_s"])}** |',
            f'| re-index, everything already cached | {indexing["resume_s"]}s '
            f'| {indexing["resume_per_file_ms"]} ms '
            f'| {_dur(indexing[f"est_resume_{target}_s"])} |',
            '',
            f'Embedder load, measured once and excluded above: '
            f'{indexing["model_load_s"]}s. The *re-index* row is what a scheduled '
            'background pass costs on a library that is already up to date — the '
            'case that runs over and over.',
        ]

    describing = payload.get('describing')
    if describing and not describing.get('unavailable'):
        lines += [
            '',
            '**Writing the descriptions** — the descriptor model, sampled per media '
            f'type ({describing["sample_per_type"]} files each). This is the real '
            'cost of indexing a fresh library, and it is measured in hours:',
            '',
            f'| Media type | s/file (mean) | median | min–max | n | '
            f'All {target:,} of this type |',
            '|---|---|---|---|---|---|',
        ]
        for name, row in describing['per_type'].items():
            if 'mean_s' not in row:
                lines.append(f'| `{name}` | — | — | — | 0 of {row["population"]} '
                             f'failed | — |')
                continue
            lines.append(
                f'| `{name}` | {row["mean_s"]} | {row["median_s"]} '
                f'| {row["min_s"]}–{row["max_s"]} | {row["sampled"]} '
                f'| {row[f"est_all_{target}_hours"]}h |')
        if describing.get('blended_per_file_s'):
            lines += [
                '',
                f'At this set\'s media mix ({describing["blended_per_file_s"]}s per '
                f'file on average), {target:,} files would take '
                f'**{describing[f"est_blended_{target}_hours"]} hours**. The mix is '
                'what varies between libraries, so the per-type rows are the ones '
                'that travel.',
                '',
                f'Descriptor load, once: {describing["model_load_s"]}s.',
            ]
    elif describing and describing.get('unavailable'):
        lines += ['', f'_Descriptor timings unavailable: '
                  f'{describing["unavailable"]}_']

    lines += [
        '',
        '**Searching**, once the files are known:',
        '',
        '| Mode | Indexed | Query (constant) | Score cold | Score warm (avg 5) '
        f'| Cold µs/file | Warm µs/file | Est. cold @{target:,} | Est. warm @{target:,} |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for row in payload['speed']:
        lines.append(
            f'| `{row["mode"]}` | {row["indexed_hits"]}/{row["files"]} '
            f'| {row["query_s"]}s | {row["cold_score_s"]}s '
            f'| {row["warm_score_s"]}s ±{row["warm_score_stdev_s"]} '
            f'| {row["cold_per_file_us"]} | {row["warm_per_file_us"]} '
            f'| {row[f"est_cold_{target}_s"]}s | {row[f"est_warm_{target}_s"]}s |'
        )

    if payload.get('quality'):
        lines += ['', '| Mode | recall@1 | recall@5 | MRR | sampled |',
                  '|---|---|---|---|---|']
        for mode, q in payload['quality'].items():
            lines.append(f'| `{mode}` | {q["recall_at_1"]} | {q["recall_at_5"]} '
                         f'| {q["mrr"]} | {q["sampled"]} |')
    else:
        lines += ['', '_Quality not measured in this run._']

    warmups = ', '.join(f'{r["mode"]} {r["model_warmup_s"]}s' for r in payload['speed'])
    lines += ['', f'Model warm-up, excluded from the timings above: {warmups}.',
              f'Queries used: ' + ', '.join(f'`{m}`: "{q}"'
                                           for m, q in payload['queries'].items()) + '.', '']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--version', required=True)
    ap.add_argument('--machine', required=True)
    ap.add_argument('--note', default=None)
    ap.add_argument('--readme', default=os.path.join(HERE, 'README.md'))
    ap.add_argument('--stdout', action='store_true', help='print instead of appending')
    args = ap.parse_args()

    with open(args.results) as fh:
        payload = json.load(fh)
    block = render(payload, args.version, args.machine, args.note)

    if args.stdout:
        print(block)
        return
    with open(args.readme, 'a', encoding='utf-8') as fh:
        fh.write('\n' + block)
    print(f'appended {args.version} to {args.readme}')


if __name__ == '__main__':
    main()
