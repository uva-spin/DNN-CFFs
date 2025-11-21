import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot kinematic values by set.")
    parser.add_argument(
        "--sets",
        type=int,
        nargs="+",
        help="Optional list of set IDs to include (space-separated).",
    )
    parser.add_argument(
        "--input",
        default="km15_bkm10_dropped.csv",
        help="Path to the CSV containing the kinematic data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    # Get unique values for each kinematic set
    kinematic_data = df.groupby("set")[["k", "QQ", "x_b", "t"]].first().reset_index()

    if args.sets:
        missing = sorted(set(args.sets) - set(kinematic_data["set"]))
        if missing:
            print(f"Warning: sets not found in data and will be ignored: {missing}")
        kinematic_data = kinematic_data[kinematic_data["set"].isin(args.sets)]
        if kinematic_data.empty:
            raise SystemExit("No matching sets found; aborting plots.")

    parameters = ["k", "QQ", "x_b", "t"]
    colors = ["blue", "orange", "green", "red"]

    suffix = (
        f"_sets_{'-'.join(str(s) for s in sorted(args.sets))}"
        if args.sets
        else ""
    )

    for param, color in zip(parameters, colors):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            kinematic_data["set"],
            kinematic_data[param],
            alpha=0.6,
            s=50,
            color=color,
        )
        plt.xlabel("Kinematic Set Number")
        plt.ylabel(param)
        plt.title(f"{param} vs Kinematic Set" + (" (filtered)" if args.sets else ""))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f"{param}_by_set{suffix}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {filename}")

    print(f"\nTotal number of kinematic sets plotted: {len(kinematic_data)}")


if __name__ == "__main__":
    main()

