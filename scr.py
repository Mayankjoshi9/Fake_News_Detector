import subprocess

# Get a list of outdated packages
result = subprocess.run(['pip', 'list', '--outdated', '--format=columns'], capture_output=True, text=True)

# Parse the output and upgrade each package
lines = result.stdout.split('\n')[2:]  # Skip the header
for line in lines:
    if line.strip():  # Ignore empty lines
        package_name = line.split()[0]
        subprocess.run(['pip', 'install', '--upgrade', package_name])
