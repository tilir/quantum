import sys

def validate_filename(filename):
    """Validate output filename and ensure .png extension"""
    if not filename:
        print("Error: Output filename cannot be empty", file=sys.stderr)
        sys.exit(1)
    return filename if filename.lower().endswith('.png') else f"{filename}.png"
