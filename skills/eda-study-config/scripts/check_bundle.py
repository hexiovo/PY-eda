"""Read-only first-stage checks. Does not certify data, methods or execution readiness."""
import argparse
import json
import math
from pathlib import Path
import sys


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonstandard JSON number: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                       object_pairs_hook=pairs, parse_constant=constant)
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    if not isinstance(value, dict):
        raise ValueError("root must be an object")
    return value


def check(config, log=None):
    errors, warnings = [], []
    required = {"schema_version", "pipeline_id", "preset_version", "input", "mapping", "qc",
                "preprocess", "decomposition", "scr", "windows", "features", "synchrony",
                "statistics", "compute", "output", "workspace", "runtime"}
    if not required <= config.keys():
        errors.append("full resolved config required; missing: " + ", ".join(sorted(required - config.keys())))
    for section in required - {"schema_version", "pipeline_id", "preset_version"}:
        if not isinstance(config.get(section), dict):
            errors.append(f"{section} must be an object")
    if errors:
        return errors, warnings
    if set(config) - required:
        errors.append("unknown top-level fields; keep research metadata in decision-log.json")
    if not config["input"].get("manifest") and not config["input"].get("signals"):
        errors.append("no input manifest/signals")
    if not config["output"].get("directory") or config["output"].get("overwrite") is not False:
        errors.append("explicit output directory and overwrite=false required")
    ids = config["features"].get("ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        errors.append("explicit nonempty feature IDs required")
        ids = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate feature IDs")
    stat = config["statistics"]
    if stat.get("enabled"):
        designs = stat.get("designs")
        if not isinstance(designs, list) or not designs:
            errors.append("inline statistics.designs required for this skill")
            designs = []
        if config["input"].get("statistical_design"):
            errors.append("do not mix inline designs with input.statistical_design")
        names = set()
        expected = {}
        for design in designs:
            if not isinstance(design, dict):
                errors.append("each design must be an object")
                continue
            name = design.get("design_id")
            if not isinstance(name, str) or not name or name in names:
                errors.append("missing/duplicate design_id")
                continue
            names.add(name)
            for key in ("outcome", "design_type", "independent_unit", "correction_family", "aggregate", "alternative"):
                if not design.get(key):
                    errors.append(f"{name}: explicit {key} required")
            if design.get("outcome") not in ids:
                errors.append(f"{name}: outcome not selected in features.ids")
            if design.get("independent_unit") not in {"subject", "group", "session", "subject_id", "group_id", "session_id"}:
                errors.append(f"{name}: invalid independent unit")
            if design.get("aggregate") not in {"mean", "median"}:
                errors.append(f"{name}: aggregate must be mean/median")
            if design.get("alternative") not in {"two-sided", "greater", "less"}:
                errors.append(f"{name}: invalid alternative")
            kind = design.get("design_type")
            if kind not in {"descriptive", "one_sample", "paired", "welch", "wilcoxon", "mannwhitney", "correlation", "mixed_effects"}:
                errors.append(f"{name}: unsupported JSON design_type")
            if "contrast" in design:
                errors.append(f"{name}: normalize contrast into explicit design fields before checking")
            levels = design.get("conditions", [])
            if kind in {"paired", "wilcoxon", "welch", "mannwhitney"}:
                key, column = ("conditions", "condition_column") if kind in {"paired", "wilcoxon"} else ("groups", "group_column")
                values = design.get(key)
                if not isinstance(values, list) or len(values) != 2 or not all(isinstance(x, str) and x for x in values) or len(set(values)) != 2:
                    errors.append(f"{name}: two distinct explicit {key} required")
                if not design.get(column):
                    errors.append(f"{name}: {column} required")
            if kind == "one_sample" and (isinstance(design.get("reference"), bool) or not isinstance(design.get("reference"), (int, float))):
                errors.append(f"{name}: numeric reference required")
            if kind == "correlation" and not design.get("predictor"):
                errors.append(f"{name}: predictor required")
            family = design.get("correction_family")
            if kind == "mixed_effects":
                if not isinstance(levels, list) or len(levels) < 2 or not all(isinstance(x, str) and x for x in levels):
                    errors.append(f"{name}: explicit mixed model conditions required")
                    levels = []
                if design.get("random_intercept") is not True or not design.get("subject_column") or not design.get("condition_column"):
                    errors.append(f"{name}: random intercept, subject_column and condition_column required")
                terms = ["Intercept"] + [f"{design.get('condition_column')}:{level}" for level in levels[1:]] + design.get("covariates", [])
                expected.update({f"{name}:{term}": family for term in terms})
            elif kind != "descriptive":
                expected[name] = family
        planned = {}
        for item in stat.get("planned_hypotheses", []):
            if not isinstance(item, dict) or not isinstance(item.get("hypothesis_id"), str) or not item.get("correction_family"):
                errors.append("planned hypotheses need explicit hypothesis_id and correction_family")
                continue
            if item["hypothesis_id"] in planned:
                errors.append("duplicate planned hypothesis")
            planned[item["hypothesis_id"]] = item["correction_family"]
        if planned != expected:
            errors.append("planned hypothesis IDs/families do not match normalized designs")
        w = config["windows"]
        kinds = int(bool(w.get("whole_record"))) + int(w.get("fixed_length_s") is not None) + int(bool(config["input"].get("events"))) + int(bool(config["input"].get("conditions")))
        if kinds > 1:
            errors.append("multiple window sources: split inferential config or document and separately verify outcome-specific filtering")
        warnings.append("static checks cannot verify independence, sample size, assumptions, CSV joins or scientific suitability")
    if config["synchrony"].get("enabled"):
        warnings.append("synchrony inference requires a separate validated adapter; surrogate probabilities are experimental")
    if log is not None:
        if log.get("unresolved"):
            errors.append("decision log contains unresolved items; draft only")
        for key in ("status", "design_sources", "hypothesis_chain", "parameter_ledger", "verification"):
            if key not in log:
                errors.append(f"decision log missing {key}")
        warnings.append("decision-log declarations are not independently verified by this helper")
    else:
        warnings.append("no decision log supplied; readiness not assessed")
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--decision-log", type=Path)
    parser.add_argument("--toolkit-root", type=Path)
    args = parser.parse_args()
    try:
        config = strict_json(args.config)
        errors, warnings = check(config, strict_json(args.decision_log) if args.decision_log else None)
        runtime_checked = False
        if args.toolkit_root:
            root = args.toolkit_root.resolve()
            if not (root / "src" / "eda_toolkit" / "config.py").is_file():
                raise ValueError("toolkit root must contain src/eda_toolkit/config.py")
            sys.path.insert(0, str(root / "src"))
            import eda_toolkit.config as runtime
            if not Path(runtime.__file__).resolve().is_relative_to(root / "src"):
                raise ValueError("loaded a different toolkit installation")
            from eda_toolkit.pipeline import resolve_run_config
            from eda_toolkit.core_features import list_feature_definitions
            resolved = resolve_run_config(args.config)
            registry = list_feature_definitions(resolved)
            supported = set(registry)
            for feature in resolved["features"]["ids"]:
                if feature not in supported:
                    errors.append(f"unknown feature ID: {feature}")
            runtime_checked = True
        else:
            warnings.append("runtime schema/feature registry not checked; use --toolkit-root or executor CLI")
        print(json.dumps({"static_valid": not errors, "runtime_schema_checked": runtime_checked,
                          "execution_ready": False, "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
        return 2 if errors else 0
    except Exception as error:
        print(json.dumps({"static_valid": False, "execution_ready": False,
                          "errors": [str(error)]}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
