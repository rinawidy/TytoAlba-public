"""
Automated Data Collection Pipeline
Runs every 15 minutes via cron:
1. Fetch data from PLN API
2. Append to historical dataset
3. Synthesize missing data points
"""
import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fetch_pln_data import main as fetch_data
from data_synthesizer import synthesize_missing_data


def main():
    """Run complete data collection pipeline"""
    print("\n" + "=" * 70)
    print("🤖 AUTOMATED DATA COLLECTION PIPELINE")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Step 1: Fetch data from PLN API
    print("STEP 1/2: Fetching PLN API Data")
    print("-" * 70)
    try:
        fetch_data()
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print("⚠️  Continuing to synthesis step anyway...")

    print("\n")

    # Step 2: Synthesize missing data
    print("STEP 2/2: Synthesizing Missing Data")
    print("-" * 70)
    try:
        synthesize_missing_data()
    except Exception as e:
        print(f"❌ Error synthesizing data: {e}")

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
