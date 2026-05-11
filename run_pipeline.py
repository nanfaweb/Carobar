import subprocess
import sys
import os
import json
import argparse
from dotenv import load_dotenv
import threading

# Load env vars from both common locations
load_dotenv()
load_dotenv("web/.env.local")

def stream_stderr(pipe):
    """Helper to print stderr in real-time from a subprocess."""
    for line in iter(pipe.readline, ''):
        if line:
            print(f"  [LOG] {line.strip()}", file=sys.stderr)

def run_command(cmd, input_data=None):
    """Runs a command, streams its stderr to console in real-time, and returns its stdout."""
    if cmd[0] == "python":
        cmd[0] = sys.executable
        
    print(f"--- Running: {' '.join(cmd)} ---")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_data else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    # Start thread to stream stderr
    stderr_thread = threading.Thread(target=stream_stderr, args=(process.stderr,))
    stderr_thread.start()

    if input_data:
        try:
            process.stdin.write(input_data)
        except BrokenPipeError:
            pass
        finally:
            process.stdin.close()

    # Read stdout manually
    stdout = ""
    if process.stdout:
        stdout = process.stdout.read()
    
    process.wait()
    stderr_thread.join()
            
    if process.returncode != 0:
        print(f"FAILED: Command exited with code {process.returncode}")
        return None
        
    return stdout



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make", type=str)
    parser.add_argument("--city", type=str)
    parser.add_argument("--pages", type=int, default=1)
    args = parser.parse_args()

    # 1. Scrape
    scrape_cmd = ["python", "scraper.py", "--pages", str(args.pages)]
    if args.make: scrape_cmd.extend(["--make", args.make])
    if args.city: scrape_cmd.extend(["--city", args.city])
    
    raw_data = run_command(scrape_cmd)
    if not raw_data: return

    # 2. Clean
    clean_cmd = ["python", "cleaner.py"]
    cleaned_data = run_command(clean_cmd, input_data=raw_data)
    if not cleaned_data: return

    # 3. Store
    store_cmd = ["python", "storer.py"]
    store_result = run_command(store_cmd, input_data=cleaned_data)
    if not store_result: return
    
    # 4. Update Embeddings
    print("--- Updating embeddings ---")
    embed_cmd = ["python", "update_embeddings.py"]
    run_command(embed_cmd)
    print("Pipeline finished successfully.")



if __name__ == "__main__":
    main()
