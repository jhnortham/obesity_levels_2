"""

Create Cron Job for Automated Updates

"""

# Import packages
import subprocess
import sys
from datetime import datetime

scripts = [
    "data_import.py",
    "etl.py",
    "model.py",
    "tableau_output.py"
]

# create datetime stamp for start of cron job
print(f"Pipeline started at: {datetime.now()}")

# create loop to run scripts

for script in scripts:
    subprocess.run([sys.executable, script], check=True)

# create datetime stamp for end of cron job

print(f"Pipeline completed at: {datetime.now()}")


