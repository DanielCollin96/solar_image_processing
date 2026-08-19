"""
Folder-based coverage check for a full year (default 2026).

For each day-folder it reads the HOUR field straight from the filenames
(``...T HH 5959Z...``) -- it does NOT open FITS or read t_obs. A day is
"complete" when all 24 hour slots 00..23 appear among its filenames. This
makes the check fast and completely agnostic to the day-boundary filing
convention (whether a folder's first frame is labelled (D-1)T235959Z as in
the 2026 data, or D T005959Z as in the older 2025 data) -- both yield 24
distinct hour slots and count as complete.

Reports, per channel:
  * incomplete day-folders  -> which hour slots are missing (+ file count)
  * entirely missing days    -> folders absent within the data's date range

Then a cross-channel diagnosis of every missing (date, hour):
  * missing on ALL channels  -> almost certainly a JSOC / upstream gap
                                (eclipse, calibration, quality flag, outage)
  * missing on SOME channels -> more likely a download-side miss for that
                                channel/run (worth re-fetching)

Usage
-----
    uv run python scripts/check_year_coverage.py
    uv run python scripts/check_year_coverage.py --year 2026
    uv run python scripts/check_year_coverage.py --channels aia_171
"""

import argparse
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from solar_image_processing.utils.pipeline_config import PipelineConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.yaml"

# Two filename conventions appear in this archive; both must be parsed.
#
#   A) aia_lev1_171a_2015_05_06t21_00_08_50z_image_lev1.fits
#         ..._YYYY_MM_DD t HH _MM _SS _ss z...      -> hour 21
#   B) aia.lev1_euv_12s.2026-04-12T005959Z.171.image.fits
#         ...YYYY-MM-DD T HHMMSS Z...               -> hour 00
#
# Older years (e.g. 2015) mix both; matching only one silently drops every
# file in the other convention and reports the whole year as "missing".
_HOUR_A = re.compile(r"_\d{4}_\d{2}_\d{2}[tT](\d{2})_\d{2}_\d{2}")
_HOUR_B = re.compile(r"\d{4}-\d{2}-\d{2}[tT](\d{2})\d{4}[zZ]")

ALL_HOURS = set(range(24))


def _file_hour(name: str):
    """Return the hour encoded in a FITS filename, or None if unrecognised.

    Tries both naming conventions found in the archive.
    """
    m = _HOUR_B.search(name) or _HOUR_A.search(name)
    if not m:
        return None
    hour = int(m.group(1))
    return hour if 0 <= hour <= 23 else None


def _file_format(name: str):
    """Return 'B', 'A', or None -- which naming convention a filename uses."""
    if _HOUR_B.search(name):
        return "B"
    if _HOUR_A.search(name):
        return "A"
    return None


def _wl_dir(base: Path, channel: str) -> Path:
    return base / "AIA" / channel.split("_")[-1]


def _folder_hours(day_dir: Path):
    """Scan one day-folder's filenames.

    Returns (hours_present, n_fits, n_unparsed, formats_seen) where
    n_unparsed counts .fits files matching NEITHER naming convention --
    a nonzero value means the parser is missing a format and the gap
    report for that day cannot be trusted.
    """
    hours, n, unparsed, formats = set(), 0, 0, set()
    for f in day_dir.glob("*.fits"):
        n += 1
        hour = _file_hour(f.name)
        if hour is None:
            unparsed += 1
            continue
        hours.add(hour)
        formats.add(_file_format(f.name))
    return hours, n, unparsed, formats


def scan_channel(base: Path, channel: str, year: int) -> dict:
    """Return {date: (hours_present, n_fits)} for every day-folder in *year*."""
    root = _wl_dir(base, channel) / str(year)
    present = {}
    if not root.exists():
        return present
    for mdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        for ddir in sorted(p for p in mdir.iterdir() if p.is_dir() and p.name.isdigit()):
            try:
                d = date(year, int(mdir.name), int(ddir.name))
            except ValueError:
                continue
            present[d] = _folder_hours(ddir)
    return present


def channel_gaps(present: dict) -> dict:
    """Return {date: set_of_missing_hours} across the channel's date range.

    Days whose folder is entirely absent (within first..last) are reported
    as all 24 hours missing.
    """
    gaps = {}
    if not present:
        return gaps
    days = sorted(present)
    d, last = days[0], days[-1]
    while d <= last:
        if d not in present:
            gaps[d] = set(ALL_HOURS)              # whole day absent
        else:
            hours = present[d][0]
            missing = ALL_HOURS - hours
            if missing:
                gaps[d] = missing
        d += timedelta(days=1)
    return gaps


def _fmt_hours(hours: set) -> str:
    if hours == ALL_HOURS:
        return "WHOLE DAY"
    return ",".join(f"{h:02d}" for h in sorted(hours))


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-year folder-based AIA coverage check.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--channels", nargs="*", default=None)
    args = parser.parse_args()

    config = PipelineConfig(CONFIG_PATH)
    base = config.paths["unprocessed"]
    channels = args.channels or config.channels

    print(f"Folder-based coverage check for {args.year} (24 filename-hours per day)")
    print(f"Base: {base}\n")

    all_gaps = {}          # channel -> {date: missing_hours}
    for channel in channels:
        present = scan_channel(base, channel, args.year)
        gaps = channel_gaps(present)
        all_gaps[channel] = gaps

        if not present:
            print(f"[{channel}] no {args.year} data found\n")
            continue

        days = sorted(present)
        total_missing = sum(len(h) for h in gaps.values())
        total_unparsed = sum(v[2] for v in present.values())
        fmts = sorted({f for v in present.values() for f in v[3] if f})

        print(f"[{channel}] {days[0]} .. {days[-1]}  |  "
              f"{len(present)} day-folders  |  "
              f"{total_missing} missing hour(s) across {len(gaps)} day(s)")
        print(f"    naming convention(s) seen: {', '.join(fmts) if fmts else 'none'}")
        if total_unparsed:
            print(f"    !! {total_unparsed} .fits file(s) matched NO known naming "
                  f"convention -- gap counts below are NOT reliable until the "
                  f"parser is extended")
        for d in sorted(gaps):
            n_files = present[d][1] if d in present else 0
            note = "" if d in present else "  (folder absent)"
            print(f"    {d}  [{len(gaps[d])}h]{note}  missing: {_fmt_hours(gaps[d])}"
                  + (f"   (files on disk: {n_files})" if d in present else ""))
        print()

    # --- cross-channel diagnosis ---
    if len(channels) > 1:
        per_slot = defaultdict(set)   # (date, hour) -> set of channels missing it
        for channel, gaps in all_gaps.items():
            for d, hours in gaps.items():
                for h in hours:
                    per_slot[(d, h)].add(channel)

        common = sorted(k for k, v in per_slot.items() if len(v) == len(channels))
        partial = sorted(k for k, v in per_slot.items() if 1 <= len(v) < len(channels))

        print("=" * 60)
        print("CROSS-CHANNEL DIAGNOSIS")
        print(f"  Missing on ALL {len(channels)} channels (likely JSOC/upstream): "
              f"{len(common)} slot(s)")
        print(f"  Missing on SOME channels (likely download-side): {len(partial)} slot(s)")
        if partial:
            print("\n  Channel-specific gaps worth a closer look:")
            by_day = defaultdict(list)
            for (d, h) in partial:
                chans = ",".join(sorted(c.split('_')[-1] for c in per_slot[(d, h)]))
                by_day[d].append(f"{h:02d}[{chans}]")
            for d in sorted(by_day):
                print(f"    {d}: {' '.join(by_day[d])}")
        print()


if __name__ == "__main__":
    main()
