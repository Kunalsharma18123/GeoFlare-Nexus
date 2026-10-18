from pathlib import Path
import json
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TILE_DIR = PROJECT_ROOT / "backend" / "data" / "osm" / "india_tiles"
OUTPUT_FILE = PROJECT_ROOT / "backend" / "data" / "osm" / "osm_context_points.csv"


def classify_feature(tags):
    if not tags:
        return None

    if (
        tags.get("landuse") == "industrial"
        or tags.get("man_made") == "works"
        or tags.get("industrial") in {"oil", "gas"}
    ):
        return "industrial"

    if tags.get("power") in {"plant", "generator"}:
        return "power"

    if (
        tags.get("industrial") in {"oil", "gas"}
        or tags.get("man_made") == "petroleum_well"
    ):
        return "oil_gas"

    if tags.get("highway") in {
        "motorway",
        "trunk",
        "primary",
        "secondary",
    }:
        return "road"

    if tags.get("place") in {"city", "town", "village"}:
        return "settlement"

    return None


def get_coordinates(element):
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]

    center = element.get("center")

    if center and "lat" in center and "lon" in center:
        return center["lat"], center["lon"]

    return None, None


def main():
    if not TILE_DIR.exists():
        raise FileNotFoundError(f"Tile directory not found: {TILE_DIR}")

    tile_files = sorted(TILE_DIR.glob("*.json"))

    print(f"Found {len(tile_files)} OSM tile files.")

    if not tile_files:
        raise RuntimeError("No OSM JSON tiles found.")

    features = {}
    processed_elements = 0
    skipped_elements = 0
    category_counts = {}

    for index, tile_file in enumerate(tile_files, start=1):
        print(f"[{index}/{len(tile_files)}] Processing {tile_file.name}")

        try:
            with tile_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            print(f"  Failed to read tile: {exc}")
            continue

        elements = data.get("elements", [])

        for element in elements:
            processed_elements += 1

            osm_type = element.get("type")
            osm_id = element.get("id")

            if osm_type is None or osm_id is None:
                skipped_elements += 1
                continue

            tags = element.get("tags", {})
            category = classify_feature(tags)

            if category is None:
                continue

            latitude, longitude = get_coordinates(element)

            if latitude is None or longitude is None:
                skipped_elements += 1
                continue

            key = (osm_type, osm_id)

            if key in features:
                continue

            name = tags.get("name", "")

            features[key] = {
                "osm_type": osm_type,
                "osm_id": osm_id,
                "category": category,
                "latitude": latitude,
                "longitude": longitude,
                "name": name,
            }

            category_counts[category] = category_counts.get(category, 0) + 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "osm_type",
            "osm_id",
            "category",
            "latitude",
            "longitude",
            "name",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features.values())

    print()
    print("OSM merge completed.")
    print(f"Tiles processed: {len(tile_files)}")
    print(f"Elements processed: {processed_elements}")
    print(f"Elements skipped: {skipped_elements}")
    print(f"Unique contextual features: {len(features)}")
    print()
    print("Feature counts:")

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()