import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
website_dir = os.path.join(current_script_dir, "Website")
os.chdir(website_dir)
import sys

if website_dir not in sys.path:
    sys.path.append(website_dir)

from interface import app


if __name__ == "__main__":
    print(f"Server starting...")
    app.run(host="127.0.0.1", port=8000, debug=False)