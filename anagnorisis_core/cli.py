"""The `anagnorisis` command.

Argparse over :mod:`anagnorisis_core.api` and nothing else — every command here
is one call into a function the Flask application also calls, so the two cannot
diverge.

    anagnorisis describe /data --sink meta            # one pass, then exit
    anagnorisis describe /data --sink meta --watch    # keep going

Describing a shared folder for the data server is exactly the second form.
"""
import argparse
import os
import signal
import sys

from anagnorisis_core import api
from anagnorisis_core.config import (SETTABLE_KEYS, describe_settings,
                                     load_config, set_user_setting,
                                     unset_user_setting, user_config_path)
from anagnorisis_core.progress import PrintProgress
from anagnorisis_core.storage.sinks import build_sink


def _add_common(parser):
    parser.add_argument('paths', nargs='+', help='files or folders to work on')
    parser.add_argument('--config', action='append', default=[],
                        help='YAML config file; repeatable, later files win. '
                             'Optional — the package ships working defaults')
    parser.add_argument('--project-config', default=None,
                        help='one folder holding cache, memory and the trained '
                             'evaluator, laid out like the application\'s '
                             'project_config/ — point it at one to share '
                             'everything with the app (default: '
                             '~/.local/share/anagnorisis/project_config)')
    parser.add_argument('--models', default=None,
                        help='downloaded model weights, kept outside the project '
                             'folder so several users can share one copy '
                             '(default: ~/.local/share/anagnorisis/models)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='do not descend into subfolders')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='files per embed/describe cycle. Each cycle costs one '
                             'load of each model, so bigger is faster; smaller '
                             'finishes .meta files sooner. 0 means do not '
                             'interleave at all — embed everything, then describe '
                             'everything (default: %(default)s)')


def _build_parser():
    parser = argparse.ArgumentParser(
        prog='anagnorisis',
        description='Describe, index and search your files.')
    sub = parser.add_subparsers(dest='command', required=True)

    describe = sub.add_parser(
        'describe',
        help='write a description for files that have none',
        description=(
            'Describes each file with the omni model and stores the result. '
            'By default it goes into the cache and nothing outside it is '
            'touched. With --sink meta the description is instead written to '
            '<file>.meta next to the file, which is how a folder carries its '
            'descriptions to another machine; an existing sidecar is never '
            'overwritten, so to regenerate one, delete it and run again.'))
    _add_common(describe)
    describe.add_argument('--sink', choices=('cache', 'meta'), default='cache',
                          help='where descriptions go (default: %(default)s). '
                               'cache keeps them to itself; meta writes a '
                               '<file>.meta sidecar into your media folders')
    describe.add_argument('--watch', action='store_true',
                          help='keep running, describing new files as they appear')
    describe.add_argument('--interval', type=int, default=600,
                          help='seconds between passes when watching (default: %(default)s)')
    describe.add_argument('--no-stamp', action='store_true',
                          help='omit the provenance line from written sidecars')
    index_p = sub.add_parser(
        'index',
        help='embed files so they can be searched',
        description=(
            'Builds the two search indexes: one from each file\'s content, one '
            'from its text description. Safe to stop at any time — progress is '
            'saved per file as it goes, and running it again resumes.'))
    _add_common(index_p)
    index_p.add_argument('--no-content', action='store_true',
                         help='skip content embeddings (semantic search)')
    index_p.add_argument('--no-metadata', action='store_true',
                         help='skip description embeddings (metadata search)')

    search_p = sub.add_parser(
        'search',
        help='find files',
        description=(
            'name     fuzzy match on the filename; needs no index.\n'
            'semantic compares your query with the file\'s own content.\n'
            'metadata compares your query with the file\'s description: tags, '
            'the model\'s sentences, internal metadata and any .meta sidecar.'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    search_p.add_argument('query')
    _add_common(search_p)
    search_p.add_argument('--mode', choices=api.SEARCH_MODES, default='metadata',
                          help='how to match (default: %(default)s)')
    search_p.add_argument('--limit', type=int, default=20)

    rate_p = sub.add_parser(
        'rate',
        help='record what you think of a file',
        description=(
            'Writes a memory file holding your rating and the file\'s '
            'description. Memory files are what the evaluator trains on, and '
            'they survive the file being renamed, moved or deleted.'))
    rate_p.add_argument('file', nargs='?',
                        help='the file to rate; omit when using --text')
    rate_p.add_argument('rating', type=float, help='your score, 0 to 10')
    rate_p.add_argument('--text', default=None,
                        help='rate this text instead of a file; - reads stdin')
    rate_p.add_argument('--memory', default=None,
                        help='the memory folder (default: from the config, '
                             '~/.local/share/anagnorisis/memory)')
    rate_p.add_argument('--describe', action='store_true',
                        help='run the descriptor now, so the memory includes it')
    rate_p.add_argument('--config', action='append', default=[])
    rate_p.add_argument('--project-config', default=None)
    rate_p.add_argument('--models', default=None)

    sort_p = sub.add_parser(
        'sort',
        help='list files by rating, best first',
        description=(
            'Your own ratings come from the memory files. With --predicted the '
            'trained evaluator fills in the files you have not rated; your '
            'ratings always take precedence over its guesses.'))
    _add_common(sort_p)
    sort_p.add_argument('--memory', default=None,
                        help='the memory folder (default: from the config)')
    sort_p.add_argument('--predicted', action='store_true',
                        help='also score unrated files with the evaluator')
    sort_p.add_argument('--limit', type=int, default=None)
    sort_p.add_argument('--personal-models', default=None,
                        help='where the trained evaluator lives (default: --models)')

    train = sub.add_parser(
        'train',
        help='train the evaluator on your ratings',
        description=(
            'Learns to predict your ratings from the memory files written when '
            'you rate something. Reads project_config/memory and writes a model; '
            'needs no database.'))
    train.add_argument('--config', action='append', default=[],
                       help='YAML config file; repeatable, later files win')
    train.add_argument('--project-config', default=None)
    train.add_argument('--models', default=None)
    train.add_argument('--memory', default=None,
                       help='the memory folder (default: from the config). Point '
                            'it at project_config/memory to train on the ratings '
                            'you made in the app')
    train.add_argument('--max-steps', type=int, default=None)
    train.add_argument('--time-budget', type=int, default=None,
                       help='stop after this many seconds')

    score_p = sub.add_parser(
        'score',
        help='ask the model what it would rate a file, or some text',
        description=(
            'Prints the rating the trained evaluator predicts. Give it a file '
            'and it scores that file\'s description; give it --text and it '
            'scores the words themselves. Short text is scored as it stands, '
            'long text is summarised first — the same rule that decides what '
            'gets stored when you rate it, so the two always agree.'))
    score_p.add_argument('file', nargs='?', help='the file to score')
    score_p.add_argument('--text', default=None,
                         help='score this text instead of a file; - reads stdin')
    score_p.add_argument('--config', action='append', default=[])
    score_p.add_argument('--project-config', default=None)
    score_p.add_argument('--models', default=None)

    config_p = sub.add_parser(
        'config',
        help='read and change where things are kept',
        description=(
            'Settings live in ~/.config/anagnorisis/config.yaml — outside the '
            'installed package, so upgrading does not overwrite them. Only the '
            'keys listed by `config list` can be set; the model names and '
            'generation settings belong to a project and are passed with '
            '--config.'))
    config_p.add_argument(
        'action', choices=('list', 'get', 'set', 'unset', 'path'),
        help='list: every setting, its value and where it came from')
    config_p.add_argument('key', nargs='?')
    config_p.add_argument('value', nargs='?')
    # The same path flags the other commands take, so `config list` can answer
    # "what would this invocation actually use?" rather than only "what is
    # stored?" — the two differ exactly when something is not taking effect.
    config_p.add_argument('--config', action='append', default=[])
    config_p.add_argument('--project-config', default=None)
    config_p.add_argument('--models', default=None)

    help_p = sub.add_parser('help', help='show help for a command')
    help_p.add_argument('topic', nargs='?', help='the command to explain')

    return parser


def _reorder_help(argv, parser):
    """Make `prog -h describe` mean what it looks like it means.

    argparse hands `-h` to whichever parser sees it first, so a help flag before
    the subcommand prints the top-level help and exits — the one page the user
    was explicitly trying to get past. Nothing in argparse fixes that from the
    inside, so the argument list is rewritten before it is parsed.

    Also accepts `prog help describe`, which is the form people try when the flag
    does not work.
    """
    commands = set(parser._subparsers._group_actions[0].choices)
    argv = list(argv)

    if argv and argv[0] == 'help':
        topic = argv[1] if len(argv) > 1 else None
        return [topic, '--help'] if topic in commands else ['--help']

    flags = {'-h', '--help'}
    if any(a in flags for a in argv):
        topic = next((a for a in argv if a in commands), None)
        if topic:
            return [topic, '--help']
    return argv


def _one_target(args, command):
    """Either a file or some text, never both and never neither.

    Both commands take the same pair, because rating a paragraph and rating a
    .txt are the same act — the model's subject is descriptions, not files.
    Returns ('file' | 'text', value), or None after printing what was wrong.
    """
    if args.text is not None and args.file:
        print(f'anagnorisis: give `{command}` a file or --text, not both.',
              file=sys.stderr)
        return None
    if args.text is not None:
        return 'text', _read_text(args.text)
    if args.file:
        return 'file', args.file
    print(f'anagnorisis: `{command}` needs a file or --text.', file=sys.stderr)
    return None


def _read_text(value: str) -> str:
    """The text argument, or standard input when it is `-`.

    Worth supporting: the interesting text is often a paragraph with newlines in
    it, which is miserable to pass as a shell argument and natural to pipe.
    """
    if value == '-':
        return sys.stdin.read()
    return value


def _handle_config(args) -> int:
    """`anag config …` — read and change where things are kept.

    Handled before the main configuration is loaded, so that `config set` still
    works when the current settings are the reason nothing else runs.
    """
    if args.action == 'path':
        print(user_config_path())
        return 0

    if args.action == 'list':
        rows = describe_settings(args.config,
                                 project_config_path=args.project_config,
                                 models_path=args.models)
        width = max(len(k) for k, _, _ in rows)
        for key, value, source in rows:
            print(f'{key:<{width}}  {value}')
            print(f'{"":<{width}}  ({source})')
        print(f'\nsettings file: {user_config_path()}'
              f'{"" if os.path.isfile(user_config_path()) else " (not created yet)"}')
        return 0

    if not args.key:
        print(f'anagnorisis: `config {args.action}` needs a setting name. '
              f'Settable: {", ".join(sorted(SETTABLE_KEYS))}', file=sys.stderr)
        return 2

    if args.action == 'get':
        rows = {k: (v, s) for k, v, s in
                describe_settings(args.config)}
        if args.key not in rows:
            print(f'anagnorisis: unknown setting {args.key!r}. '
                  f'Settable: {", ".join(sorted(SETTABLE_KEYS))}', file=sys.stderr)
            return 2
        value, source = rows[args.key]
        print(value)
        return 0

    if args.action == 'set':
        if args.value is None:
            print(f'anagnorisis: `config set {args.key}` needs a value.',
                  file=sys.stderr)
            return 2
        try:
            path = set_user_setting(args.key, args.value)
        except (ValueError, OSError) as exc:
            print(f'anagnorisis: {exc}', file=sys.stderr)
            return 2
        print(f'{args.key} = {args.value}')
        print(f'written to {path}')
        return 0

    # unset
    try:
        removed = unset_user_setting(args.key)
    except (ValueError, OSError) as exc:
        print(f'anagnorisis: {exc}', file=sys.stderr)
        return 2
    print(f'{args.key} {"cleared" if removed else "was not set"}; '
          f'the default applies.')
    return 0


def main(argv=None) -> int:
    # `anag config list | head` and `anag search … | head` are the obvious things
    # to type, and Python's default SIGPIPE handling turns the closed pipe into a
    # BrokenPipeError traceback. Restoring the default disposition makes the
    # command die quietly the way every other unix tool does.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):
        pass    # no SIGPIPE on this platform, or not on the main thread

    parser = _build_parser()
    args = parser.parse_args(_reorder_help(
        sys.argv[1:] if argv is None else argv, parser))

    if args.command == 'config':
        return _handle_config(args)

    try:
        cfg = load_config(args.config,
                          project_config_path=args.project_config,
                          models_path=args.models)
    except ValueError as exc:
        print(f"anagnorisis: {exc}", file=sys.stderr)
        return 2

    if args.command == 'describe':
        # The cache sink needs the description cache and its key formula, which
        # live in the core alongside metadata search — so both sinks work here.
        metadata_search = None
        if args.sink == 'cache':
            from anagnorisis_core.search.metadata_search import get_metadata_search
            metadata_search = get_metadata_search(cfg)

        sink = build_sink(args.sink, cfg=cfg, metadata_search=metadata_search)
        if args.no_stamp:
            sink._stamp = False
        progress = PrintProgress()

        if args.watch:
            api.watch(args.paths, cfg=cfg, sink=sink, ctx=progress,
                      interval_seconds=args.interval,
                      recursive=not args.no_recursive,
                      batch_size=args.batch_size)
            return 0

        report = api.describe(args.paths, cfg=cfg, sink=sink, ctx=progress,
                              recursive=not args.no_recursive,
                              batch_size=args.batch_size)
        progress.done(str(report))
        for err in report.errors[:10]:
            print(f"  ! {err}", file=sys.stderr)
        return 1 if report.failed and not report.written else 0

    if args.command == 'index':
        progress = PrintProgress()
        report = api.index(args.paths, cfg=cfg, ctx=progress,
                           recursive=not args.no_recursive,
                           batch_size=args.batch_size,
                           content=not args.no_content,
                           metadata=not args.no_metadata)
        progress.done(str(report))
        for err in report.errors[:10]:
            print(f'  ! {err}', file=sys.stderr)
        return 0

    if args.command == 'search':
        results = api.search(args.query, args.paths, cfg=cfg, mode=args.mode,
                             limit=args.limit,
                             recursive=not args.no_recursive)
        if not results:
            print('no matches. If you expected some, the files may not be '
                  'indexed yet — try `anagnorisis index` first.', file=sys.stderr)
            return 1
        for path, score in results:
            print(f'{score:7.4f}  {path}')
        return 0

    if args.command == 'score':
        from anagnorisis_core.models.embedder import get_omni_embedder
        target = _one_target(args, 'score')
        if target is None:
            return 2
        kind, value = target
        try:
            score = (api.score_file(value, cfg=cfg) if kind == 'file'
                     else api.score_text(value, cfg=cfg,
                                         models_folder=args.models))
            print(f'{score:.2f}')
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f'anagnorisis: {exc}', file=sys.stderr)
            return 1
        finally:
            # score_text deliberately leaves the model loaded, so a caller
            # scoring many texts pays for it once. A command line invocation
            # scores one and exits, so it releases the card on the way out.
            get_omni_embedder(cfg).unload()
        return 0

    if args.command == 'rate':
        target = _one_target(args, 'rate')
        if target is None:
            return 2
        kind, value = target
        progress = PrintProgress()
        try:
            if kind == 'text':
                written = api.rate_text(
                    value, args.rating, cfg=cfg,
                    memory_dir=args.memory or cfg.main.memory_path,
                    models_folder=args.models, ctx=progress)
                print(f'wrote {written}')
                return 0
            written = api.rate(args.file, args.rating, cfg=cfg,
                               memory_dir=args.memory or cfg.main.memory_path,
                               describe_now=args.describe, ctx=progress)
        except Exception as exc:
            progress.done('')
            print(f'anagnorisis: {exc}', file=sys.stderr)
            return 1
        progress.done(f'wrote {written}')
        return 0

    if args.command == 'sort':
        if args.personal_models:
            cfg.main.personal_models_path = args.personal_models
        try:
            rows = api.rank_by_rating(args.paths, cfg=cfg,
                                      memory_dir=args.memory or cfg.main.memory_path,
                                      predicted=args.predicted,
                                      limit=args.limit,
                                      recursive=not args.no_recursive)
        except FileNotFoundError as exc:
            print(f'anagnorisis: {exc}', file=sys.stderr)
            return 1
        if not rows:
            print('nothing rated yet. Use `anagnorisis rate`, or pass '
                  '--predicted to use the model.', file=sys.stderr)
            return 1
        for path, rating, source in rows:
            print(f'{rating:6.2f}  {source:<5}  {path}')
        return 0

    if args.command == 'train':
        if args.memory:
            cfg.main.memory_path = args.memory
        progress = PrintProgress()
        try:
            saved = api.train_evaluator(cfg=cfg, ctx=progress,
                                        max_steps=args.max_steps,
                                        time_budget_seconds=args.time_budget)
        except Exception as exc:
            progress.done('')
            print(f"anagnorisis: training failed: {exc}", file=sys.stderr)
            return 1
        progress.done(f'trained: {saved}')
        return 0

    return 2


if __name__ == '__main__':
    raise SystemExit(main())
