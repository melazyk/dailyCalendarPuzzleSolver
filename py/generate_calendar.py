#!/usr/bin/env python3
"""
Generate an interactive HTML calendar from CSV data showing puzzle solutions per day.
Reads a CSV file with Date and Solutions columns, generates a heatmap calendar.
"""

import csv
import argparse
from datetime import datetime as dt
from calendar import monthcalendar, month_name


def read_csv(csv_file):
    """Read solutions data from CSV file."""
    solutions_by_date = {}

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                date = dt.strptime(row['Date'], '%Y-%m-%d').date()
                solutions = int(row['Solutions'])
                solutions_by_date[date] = solutions
            except (ValueError, KeyError):
                continue

    return solutions_by_date


def get_color_for_count(count):
    """Return a CSS color based on solution count."""
    if count == -1:
        return "#cccccc"  # Gray for error
    elif count == 0:
        return "#ffcccc"  # Light red
    elif count <= 10:
        return "#ffeeaa"  # Light yellow
    elif count <= 50:
        return "#ccffaa"  # Light green
    elif count <= 100:
        return "#aaff99"  # Medium green
    else:
        return "#00dd00"  # Bright green


def generate_html_calendar(solutions_by_date, output_file):
    """Generate an HTML calendar with solution counts."""
    if not solutions_by_date:
        print("No data to generate calendar")
        return

    # Determine year from data
    year = min(solutions_by_date.keys()).year

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dragon Fjord Calendar Puzzle - Solutions per Day {year}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .year-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-box {{
            width: 30px;
            height: 30px;
            border: 1px solid #999;
        }}
        .months-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .month {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .month-title {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .calendar-grid {{
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 2px;
        }}
        .weekday-header {{
            text-align: center;
            font-weight: bold;
            padding: 8px 4px;
            background-color: #f0f0f0;
            border-radius: 4px;
            font-size: 12px;
        }}
        .day-cell {{
            aspect-ratio: 1;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 4px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .day-cell:hover {{
            border-color: #333;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
        }}
        .day-number {{
            font-weight: bold;
            font-size: 12px;
        }}
        .solution-count {{
            font-size: 10px;
            margin-top: 2px;
        }}
        .empty-cell {{
            background-color: #f9f9f9;
        }}
        .stats {{
            text-align: center;
            margin-top: 30px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-content {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .stat-box {{
            padding: 10px;
        }}
        .stat-label {{
            color: #666;
            font-size: 12px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="year-container">
        <h1>Dragon Fjord Calendar Puzzle - Solutions per Day {year}</h1>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-box" style="background-color: #ffcccc;"></div>
                <span>0 solutions</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background-color: #ffeeaa;"></div>
                <span>1-10 solutions</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background-color: #ccffaa;"></div>
                <span>11-50 solutions</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background-color: #aaff99;"></div>
                <span>51-100 solutions</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background-color: #00dd00;"></div>
                <span>100+ solutions</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background-color: #cccccc;"></div>
                <span>No data</span>
            </div>
        </div>
        <div class="months-grid">
"""

    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    for month in range(1, 13):
        month_name_str = month_name[month]
        html += f"""        <div class="month">
            <div class="month-title">{month_name_str} {year}</div>
            <div class="calendar-grid">
"""

        # Add weekday headers
        for day in weekdays:
            html += f'                <div class="weekday-header">{day}</div>\n'

        # Get calendar for this month
        cal = monthcalendar(year, month)

        # Add day cells
        for week in cal:
            for day in week:
                if day == 0:
                    html += '                <div class="day-cell empty-cell"></div>\n'
                else:
                    date = dt(year, month, day).date()
                    count = solutions_by_date.get(date, -1)
                    color = get_color_for_count(count)
                    count_str = str(count) if count >= 0 else "-"
                    html += f"""                <div class="day-cell" style="background-color: {color};">
                    <div class="day-number">{day}</div>
                    <div class="solution-count">{count_str}</div>
                </div>\n"""

        html += """            </div>
        </div>
"""

    # Calculate statistics
    total_days = len([c for c in solutions_by_date.values() if c >= 0])
    total_solutions = sum(c for c in solutions_by_date.values() if c >= 0)
    avg_solutions = total_solutions / total_days if total_days > 0 else 0
    max_solutions = max((c for c in solutions_by_date.values() if c >= 0), default=0)
    min_solutions = min((c for c in solutions_by_date.values() if c >= 0), default=0)
    days_with_zero = sum(1 for c in solutions_by_date.values() if c == 0)

    html += f"""        </div>
        <div class="stats">
            <h2>Statistics for {year}</h2>
            <div class="stats-content">
                <div class="stat-box">
                    <div class="stat-label">Total Days Calculated</div>
                    <div class="stat-value">{total_days}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total Solutions</div>
                    <div class="stat-value">{total_solutions}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Average Solutions/Day</div>
                    <div class="stat-value">{avg_solutions:.1f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Min Solutions in a Day</div>
                    <div class="stat-value">{min_solutions}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Max Solutions in a Day</div>
                    <div class="stat-value">{max_solutions}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Days with 0 Solutions</div>
                    <div class="stat-value">{days_with_zero}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML calendar from puzzle solutions CSV data"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input CSV file with Date and Solutions columns"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calendar.html",
        help="Output HTML file (default: calendar.html)"
    )

    args = parser.parse_args()

    print(f"Reading data from {args.input}...")
    solutions_by_date = read_csv(args.input)

    if not solutions_by_date:
        print("Error: No valid data found in CSV file")
        return

    print(f"Found {len(solutions_by_date)} days of data")

    # Generate HTML calendar
    output_path = generate_html_calendar(solutions_by_date, args.output)
    print(f"✓ HTML calendar saved to: {output_path}")
    print(f"  Open it in a browser to view the results")


if __name__ == "__main__":
    main()
