from build_clusters import build_clusters
from build_layout import build_layout
from build_spatial_chunks import build_spatial_chunks
from build_tiles import build_tiles


def main():
    # build_clusters()
    build_layout()
    build_spatial_chunks()
    build_tiles()

    print("\nPipeline finished successfully.")
    print("Now serve this folder with:")
    print("  python serve.py --port 8000")
    print("Then open:")
    print("  http://localhost:8000/viewer.html")


if __name__ == "__main__":
    main()
