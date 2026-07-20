#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Scorer Interface for Speech Evaluation."""

import argparse
import logging

from versa.metric_discovery import (
    create_metric_discovery_registry,
    describe_metric,
    format_metric_list,
    parse_metric_category,
    parse_metric_type,
    recommend_config,
    supported_recommendation_tasks,
)


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Speech Evaluation Interface")
    parser.add_argument(
        "--pred",
        type=str,
        help="Wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "--score_config", type=str, default=None, help="Configuration of Score Config"
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Wav.scp for ground truth waveforms.",
    )
    parser.add_argument(
        "--text", type=str, default=None, help="Path of ground truth transcription."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path of directory to write the results.",
    )
    parser.add_argument(
        "--cache_folder", type=str, default=None, help="Path of cache saving"
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="whether to use GPU if it can"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of local CPU worker processes for utterance scoring.",
    )
    parser.add_argument(
        "--io",
        type=str,
        default="kaldi",
        choices=["kaldi", "soundfile", "dir"],
        help="io interface to use",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    parser.add_argument(
        "--rank",
        default=0,
        type=int,
        help="the overall rank in the batch processing, used to specify GPU rank",
    )
    parser.add_argument(
        "--no_match",
        action="store_true",
        help="Do not match the groundtruth and generated files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume utterance scoring from an existing output_file by skipping "
            "keys already present in the JSONL results."
        ),
    )
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="utterance",
        choices=["utterance", "metric"],
        help=(
            "Scoring loop order. Use 'metric' to load one metric at a time, "
            "reducing peak GPU memory when many metrics are configured."
        ),
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help=(
            "Optional report path generated from utterance-level scores after "
            "scoring. Format is inferred from extension unless --report-format "
            "is set."
        ),
    )
    parser.add_argument(
        "--report-format",
        choices=["auto", "html", "csv", "md"],
        default="auto",
        help="Report format for --report.",
    )
    parser.add_argument(
        "--report-group-by",
        default=None,
        help="Optional scored record field used for per-metric report rankings.",
    )
    parser.add_argument(
        "--report-outlier-limit",
        type=int,
        default=3,
        help="Maximum outlier examples to keep per metric in --report.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List registered metrics and exit.",
    )
    parser.add_argument(
        "--describe-metric",
        type=str,
        default=None,
        metavar="NAME",
        help="Describe one metric by name or alias and exit.",
    )
    parser.add_argument(
        "--recommend-config",
        action="store_true",
        help="Print a recommended YAML score config and exit.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Task for --recommend-config. Supported tasks: "
            + ", ".join(supported_recommendation_tasks())
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Target device for --recommend-config.",
    )
    parser.add_argument(
        "--metric-category",
        type=str,
        default=None,
        choices=["independent", "dependent", "non_match", "distributional"],
        help="Filter --list-metrics by metric category.",
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        default=None,
        choices=[
            "string",
            "float",
            "int",
            "bool",
            "list",
            "dict",
            "tuple",
            "array",
            "time",
        ],
        help="Filter --list-metrics by output type.",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num_workers must be at least 1")
    if args.num_workers > 1 and args.use_gpu:
        parser.error("--num_workers > 1 is CPU-only and cannot be used with --use_gpu")

    if args.list_metrics or args.describe_metric or args.recommend_config:
        try:
            if args.list_metrics:
                registry = create_metric_discovery_registry()
                category = parse_metric_category(args.metric_category)
                metric_type = parse_metric_type(args.metric_type)
                print(format_metric_list(registry, category, metric_type))
                return
            if args.describe_metric:
                registry = create_metric_discovery_registry()
                print(describe_metric(registry, args.describe_metric))
                return
            if args.recommend_config:
                if not args.task:
                    parser.error("--recommend-config requires --task")
                print(recommend_config(args.task, args.device))
                return
        except ValueError as e:
            parser.error(str(e))

    import torch

    from versa.definition import MetricCategory
    from versa.scorer_shared import (
        audio_loader_setup,
        configure_metric_cache_dirs,
        configure_shared_cache_environment,
        VersaScorer,
        compute_summary,
    )
    import yaml

    # In case of using `local` backend, all GPU will be visible to all process.
    if args.use_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("--use_gpu was set, but no CUDA device is available")
        gpu_rank = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(gpu_rank)
        logging.info(f"using device: cuda:{gpu_rank}")

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    with open(args.score_config, "r", encoding="utf-8") as f:
        score_config = yaml.safe_load(f)
    configure_shared_cache_environment(args.cache_folder)
    score_config = configure_metric_cache_dirs(score_config, args.cache_folder)

    # Validate before any scoring or model setup begins.
    scorer = VersaScorer()
    try:
        from versa.config_validation import validate_score_config

        validate_score_config(
            score_config,
            registry=scorer.registry,
            use_gt=(args.gt is not None and args.gt != "None" and not args.no_match),
            use_gt_text=(args.text is not None),
            use_gpu=args.use_gpu,
        )
    except ValueError as e:
        parser.error(str(e))

    gen_files = audio_loader_setup(args.pred, args.io)

    # find reference file
    args.gt = None if args.gt == "None" else args.gt
    if args.gt is not None and not args.no_match:
        gt_files = audio_loader_setup(args.gt, args.io)
    else:
        gt_files = None

    # fine ground truth transcription
    if args.text is not None:
        text_info = {}
        with open(args.text) as f:
            for line in f.readlines():
                key, value = line.strip().split(maxsplit=1)
                text_info[key] = value
    else:
        text_info = None

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if gt_files is not None and len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )

    logging.info("The number of utterances = %d" % len(gen_files))

    # Initialize VersaScorer
    score_metadata = {
        config["name"]: scorer.registry.get_metadata(config["name"])
        for config in score_config
    }
    corpus_score_config = [
        config
        for config in score_config
        if (
            score_metadata[config["name"]]
            and score_metadata[config["name"]].category == MetricCategory.DISTRIBUTIONAL
        )
    ]
    utterance_score_config = [
        config
        for config in score_config
        if not (
            score_metadata[config["name"]]
            and score_metadata[config["name"]].category == MetricCategory.DISTRIBUTIONAL
        )
    ]

    if args.num_workers > 1 and args.scoring_mode == "metric":
        parser.error(
            "--num_workers > 1 is only supported with --scoring_mode utterance"
        )
    if args.num_workers > 1 and len(utterance_score_config) == 0:
        parser.error("--num_workers > 1 requires at least one utterance-level metric")

    score_info = []
    if args.scoring_mode == "metric":
        score_info = scorer.score_utterances_by_metric(
            gen_files,
            utterance_score_config,
            gt_files,
            text_info,
            output_file=args.output_file,
            io=args.io,
            resume=args.resume,
            use_gpu=args.use_gpu,
        )
        logging.info("Summary: {}".format(compute_summary(score_info)))
        utterance_metric_count = int(
            any(any(key != "key" for key in score) for score in score_info)
        )
    else:
        # Load utterance-level metrics
        utterance_metrics = scorer.load_metrics(
            utterance_score_config,
            use_gt=(True if gt_files is not None else False),
            use_gt_text=(True if text_info is not None else False),
            use_gpu=args.use_gpu,
        )

        utterance_metric_count = len(
            [
                metric
                for metric in utterance_metrics.metrics.values()
                if metric.get_metadata().category != MetricCategory.DISTRIBUTIONAL
            ]
        )

    # Perform utterance-level scoring
    if args.scoring_mode == "utterance" and len(utterance_metrics.metrics) > 0:
        score_info = scorer.score_utterances(
            gen_files,
            utterance_metrics,
            gt_files,
            text_info,
            output_file=args.output_file,
            io=args.io,
            resume=args.resume,
            num_workers=args.num_workers,
        )
        logging.info("Summary: {}".format(compute_summary(score_info)))
    elif utterance_metric_count == 0:
        logging.info("No utterance-level scoring function is provided.")

    # Load corpus-level metrics (distributional metrics)
    corpus_metrics = scorer.load_metrics(
        corpus_score_config,
        use_gt=(True if gt_files is not None else False),
        use_gt_text=(True if text_info is not None else False),
        use_gpu=args.use_gpu,
    )

    # Filter for corpus-level metrics and perform corpus scoring
    corpus_suite = corpus_metrics.filter_by_category(MetricCategory.DISTRIBUTIONAL)
    if len(corpus_suite.metrics) > 0:
        corpus_score_info = scorer.score_corpus(
            gen_files,
            corpus_suite,
            gt_files,
            text_info,
            output_file=args.output_file + ".corpus" if args.output_file else None,
        )
        logging.info("Corpus Summary: {}".format(corpus_score_info))
    else:
        logging.info("No corpus-level scoring function is provided.")

    # Ensure at least one scoring function is provided
    if utterance_metric_count == 0 and len(corpus_suite.metrics) == 0:
        raise ValueError("No scoring function is provided")

    if args.report:
        if not score_info:
            raise ValueError("--report requires at least one utterance-level score")
        _write_report(
            score_info,
            args.report,
            report_format=args.report_format,
            group_by=args.report_group_by,
            outlier_limit=args.report_outlier_limit,
            registry=scorer.registry,
        )


def _write_report(
    score_info,
    report_path,
    *,
    report_format,
    group_by,
    outlier_limit,
    registry,
):
    from pathlib import Path

    from versa.reporting import (
        analyze_records,
        write_csv_report,
        write_html_report,
        write_markdown_report,
    )

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_format == "auto":
        report_format = {
            ".html": "html",
            ".htm": "html",
            ".csv": "csv",
            ".md": "md",
            ".markdown": "md",
        }.get(output_path.suffix.lower(), "html")

    analysis = analyze_records(
        score_info,
        group_by=group_by,
        outlier_limit=outlier_limit,
        registry=registry,
    )
    if report_format == "html":
        write_html_report(analysis, report_path)
    elif report_format == "csv":
        write_csv_report(analysis, report_path)
    else:
        write_markdown_report(analysis, report_path)

    logging.info("Wrote %s report to %s", report_format, report_path)


if __name__ == "__main__":
    main()
