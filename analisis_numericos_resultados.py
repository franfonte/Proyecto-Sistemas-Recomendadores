import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_PATH = "results.json"
DATASET_ORDER = [10, 25, 50, 75, 100]
OUTPUT_DIR = Path("analisis")


def load_results(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    rows = []
    for size, models in raw.items():
        for model_name, values in models.items():
            perf = values["performance_metrics"]
            pred = values["prediction_footprint"]
            train = values["training_footprint"]
            rows.append(
                {
                    "dataset_size": int(size),
                    "model": model_name,
                    "ndcg_at_10": float(perf["ndcg_at_10"]),
                    "total_energy_kWh": float(
                        pred["energy_consumed_kWh"] + train["energy_consumed_kWh"]
                    ),
                    "training_energy_kWh": float(train["energy_consumed_kWh"]),
                    "prediction_energy_kWh": float(pred["energy_consumed_kWh"]),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "dataset_size"])
    return df


def compute_percentage_changes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset_size"] = pd.Categorical(df["dataset_size"], DATASET_ORDER, ordered=True)
    df = df.sort_values(["model", "dataset_size"])

    for column in ["ndcg_at_10", "total_energy_kWh", "training_energy_kWh", "prediction_energy_kWh"]:
        grouped = df.groupby("model")[column]
        df[f"pct_change_{column}"] = grouped.pct_change() * 100
        df[f"delta_{column}"] = grouped.diff()

    return df


def summarize_transitions(df: pd.DataFrame) -> pd.DataFrame:
    transitions = []

    for model, group in df.groupby("model"):
        group = group.sort_values("dataset_size")
        previous = None

        for _, row in group.iterrows():
            if previous is not None:
                transitions.append(
                    {
                        "model": model,
                        "from_dataset": previous["dataset_size"],
                        "to_dataset": row["dataset_size"],
                        "ndcg_delta_pct": row["pct_change_ndcg_at_10"],
                        "total_energy_delta_pct": row["pct_change_total_energy_kWh"],
                        "ndcg_delta_abs": row["delta_ndcg_at_10"],
                        "total_energy_delta_abs": row["delta_total_energy_kWh"],
                        "training_energy_delta_abs": row["delta_training_energy_kWh"],
                        "prediction_energy_delta_abs": row["delta_prediction_energy_kWh"],
                        "training_energy_delta_pct": row["pct_change_training_energy_kWh"],
                        "prediction_energy_delta_pct": row["pct_change_prediction_energy_kWh"],
                    }
                )
            previous = row

    summary_df = pd.DataFrame(transitions)
    summary_df = summary_df.sort_values(["model", "from_dataset", "to_dataset"])
    return summary_df


def df_lookup_value(df: pd.DataFrame, model: str, dataset: int, column: str) -> float:
    dataset = int(dataset)
    match = df[(df["model"] == model) & (df["dataset_size"] == dataset)][column]
    if match.empty:
        return float("nan")
    return float(match.iloc[0])


def describe_findings(summary_df: pd.DataFrame, base_df: pd.DataFrame) -> list[str]:
    pd.options.display.float_format = "{:.2f}".format

    print("=== Cambios porcentuales por modelo y transición ===")
    print(summary_df)

    print("\n=== Principales hallazgos ===")
    findings: list[str] = []

    for model, group in summary_df.groupby("model"):
        biggest_gain = group.loc[group["ndcg_delta_pct"].idxmax()]
        energy_tradeoff = biggest_gain["total_energy_delta_pct"]
        ndcg_abs_delta = (
            df_lookup_value(base_df, model, biggest_gain["to_dataset"], "ndcg_at_10")
            - df_lookup_value(base_df, model, biggest_gain["from_dataset"], "ndcg_at_10")
        )
        energy_abs_delta = (
            df_lookup_value(base_df, model, biggest_gain["to_dataset"], "total_energy_kWh")
            - df_lookup_value(base_df, model, biggest_gain["from_dataset"], "total_energy_kWh")
        )

        findings.append(f"Modelo: {model}")
        findings.append(
            "  Mayor mejora de NDCG@10: "
            f"{biggest_gain['ndcg_delta_pct']:.2f}% (Δ {ndcg_abs_delta:.4f}) al pasar de "
            f"{biggest_gain['from_dataset']}% a {biggest_gain['to_dataset']}%"
        )
        findings.append(
            "  Costo energético total asociado: "
            f"{energy_tradeoff:.2f}% (Δ {energy_abs_delta:.6f} kWh)"
        )
        print(f"\n{findings[-3]}")
        print(findings[-2])
        print(findings[-1])

        over_50 = group[(group["from_dataset"] >= 50) | (group["to_dataset"] >= 50)]
        if not over_50.empty:
            high_transition = over_50.iloc[-1]
            ndcg_high_delta = (
                df_lookup_value(base_df, model, high_transition["to_dataset"], "ndcg_at_10")
                - df_lookup_value(base_df, model, high_transition["from_dataset"], "ndcg_at_10")
            )
            energy_high_delta = (
                df_lookup_value(base_df, model, high_transition["to_dataset"], "total_energy_kWh")
                - df_lookup_value(base_df, model, high_transition["from_dataset"], "total_energy_kWh")
            )
            finding = (
                "  Último salto≥50%: "
                f"NDCG {high_transition['ndcg_delta_pct']:.2f}% (Δ {ndcg_high_delta:.4f}) | "
                f"energía {high_transition['total_energy_delta_pct']:.2f}% "
                f"(Δ {energy_high_delta:.6f} kWh)"
            )
            findings.append(finding)
            print(finding)

    return findings


def build_operational_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["dataset_size"] = data["dataset_size"].astype(int)

    data["ndcg_per_total_kWh"] = np.where(
        data["total_energy_kWh"] > 0,
        data["ndcg_at_10"] / data["total_energy_kWh"],
        np.nan,
    )
    data["ndcg_per_training_kWh"] = np.where(
        data["training_energy_kWh"] > 0,
        data["ndcg_at_10"] / data["training_energy_kWh"],
        np.nan,
    )
    data["ndcg_per_prediction_kWh"] = np.where(
        data["prediction_energy_kWh"] > 0,
        data["ndcg_at_10"] / data["prediction_energy_kWh"],
        np.nan,
    )
    data["training_share"] = np.where(
        data["total_energy_kWh"] > 0,
        data["training_energy_kWh"] / data["total_energy_kWh"],
        np.nan,
    )
    data["prediction_share"] = np.where(
        data["total_energy_kWh"] > 0,
        data["prediction_energy_kWh"] / data["total_energy_kWh"],
        np.nan,
    )

    return data


def build_marginal_efficiency(summary_df: pd.DataFrame) -> pd.DataFrame:
    data = summary_df.copy()
    data["ndcg_gain_per_total_kWh"] = np.where(
        data["total_energy_delta_abs"].abs() > 0,
        data["ndcg_delta_abs"] / data["total_energy_delta_abs"],
        np.nan,
    )
    data["ndcg_gain_per_training_kWh"] = np.where(
        data["training_energy_delta_abs"].abs() > 0,
        data["ndcg_delta_abs"] / data["training_energy_delta_abs"],
        np.nan,
    )
    data["ndcg_gain_per_prediction_kWh"] = np.where(
        data["prediction_energy_delta_abs"].abs() > 0,
        data["ndcg_delta_abs"] / data["prediction_energy_delta_abs"],
        np.nan,
    )
    return data


def describe_efficiency(
    operational_df: pd.DataFrame, marginal_df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    print("\n=== Eficiencia energética y costo operacional ===")

    operational_lines: list[str] = ["=== Eficiencia operativa (NDCG@10 por kWh) ==="]
    for model, group in operational_df.groupby("model"):
        subset = group[group["dataset_size"] == 100]
        if subset.empty:
            msg = "  No hay observaciones para el 100% del dataset."
            line = f"{model}: {msg}"
            operational_lines.append(line)
            print(line)
        else:
            row = subset.iloc[0]
            msg = (
                f"  NDCG/kWh total: {row['ndcg_per_total_kWh']:.4f} | "
                f"entrenamiento {row['training_share']*100:.1f}% | "
                f"predicción {row['prediction_share']*100:.1f}%"
            )
            line = f"{model}: {msg}"
            operational_lines.append(line)
            print(line)

    marginal_lines: list[str] = ["=== Eficiencia marginal (ganancia NDCG/kWh) ==="]
    for model, group in marginal_df.groupby("model"):
        valid = group[np.isfinite(group["ndcg_gain_per_total_kWh"])]
        if valid.empty:
            msg = "  No se pudo calcular eficiencia marginal (división por cero)."
            line = f"{model}: {msg}"
            marginal_lines.append(line)
            print(line)
        else:
            best = valid.sort_values("ndcg_gain_per_total_kWh", ascending=False).iloc[0]
            worst = valid.sort_values("ndcg_gain_per_total_kWh").iloc[0]
            best_msg = (
                f"  Mejor salto: {int(best['from_dataset'])}%→{int(best['to_dataset'])}% | "
                f"{best['ndcg_gain_per_total_kWh']:.4f} NDCG/kWh"
            )
            worst_msg = (
                f"  Peor salto: {int(worst['from_dataset'])}%→{int(worst['to_dataset'])}% | "
                f"{worst['ndcg_gain_per_total_kWh']:.4f} NDCG/kWh"
            )
            best_line = f"{model}: {best_msg}"
            worst_line = f"{model}: {worst_msg}"
            marginal_lines.extend([best_line, worst_line])
            print(best_line)
            print(worst_line)

    return operational_lines, marginal_lines


def export_tables(
    summary_df: pd.DataFrame,
    findings: list[str],
    operational_df: pd.DataFrame,
    marginal_df: pd.DataFrame,
    operational_lines: list[str],
    marginal_lines: list[str],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(OUTPUT_DIR / "transiciones_porcentuales.csv", index=False)
    operational_df.to_csv(OUTPUT_DIR / "eficiencia_operacional.csv", index=False)
    marginal_df.to_csv(OUTPUT_DIR / "eficiencia_marginal.csv", index=False)

    with open(OUTPUT_DIR / "principales_hallazgos.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(findings))

    with open(OUTPUT_DIR / "eficiencia_operacional.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(operational_lines))

    with open(OUTPUT_DIR / "eficiencia_marginal.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(marginal_lines))


def main() -> None:
    df = load_results(RESULTS_PATH)
    df = compute_percentage_changes(df)
    summary_df = summarize_transitions(df)
    findings = describe_findings(summary_df, df)
    operational_df = build_operational_efficiency(df)
    marginal_df = build_marginal_efficiency(summary_df)
    operational_lines, marginal_lines = describe_efficiency(operational_df, marginal_df)
    export_tables(
        summary_df,
        findings,
        operational_df,
        marginal_df,
        operational_lines,
        marginal_lines,
    )


if __name__ == "__main__":
    main()
