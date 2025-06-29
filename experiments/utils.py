import sys


def validate_filename(filename, fileext):
    """Validate output filename and ensure .{fileext} extension"""
    if not filename:
        print("Error: Output filename cannot be empty", file=sys.stderr)
        sys.exit(1)
    if not fileext:
        print("Error: Output fileext cannot be empty", file=sys.stderr)
        sys.exit(1)
    return (
        filename
        if filename.lower().endswith(f".{fileext}")
        else f"{filename}.{fileext}"
    )
