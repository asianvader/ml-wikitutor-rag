from pathlib import Path

INPUT = Path("data_raw/titles.txt")
OUTPUT = Path("data_raw/titles_deduped.txt")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found")

    lines = [
        line.strip()
        for line in INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    unique = sorted(set(lines), key=str.lower)

    OUTPUT.write_text("\n".join(unique) + "\n", encoding="utf-8")

    print(f"Original count: {len(lines)}")
    print(f"Unique count:   {len(unique)}")
    print(f"Removed:        {len(lines) - len(unique)}")
    print(f"Wrote: {OUTPUT}")


if __name__ == "__main__":
    main()