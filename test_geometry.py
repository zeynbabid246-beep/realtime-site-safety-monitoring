from src.safety.geometry import (
    extract_cone_points,
    cluster_cones,
    create_polygon_from_points,
    build_danger_zones,
)


def main():

    # Fake Safety Cone detections
    # [x1, y1, x2, y2, confidence, class_id]
    cones = [
        [100, 100, 200, 200, 0.90, 6],
        [500, 100, 600, 200, 0.85, 6],
        [600, 400, 700, 500, 0.88, 6],
        [100, 400, 200, 500, 0.92, 6],
    ]

    print("=== GEOMETRY TEST ===")

    # 1. Extract cone points
    points = extract_cone_points(cones)

    print("\n1. Cone points:")
    print(points)

    # 2. Cluster cones
    clusters = cluster_cones(
        points,
        min_cluster_size=3,
        min_samples=2,
    )

    print("\n2. Cone clusters:")
    print("Number of clusters:", len(clusters))

    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {cluster}")

    # 3. Create polygons
    polygons = build_danger_zones(
        cones,
        min_cluster_size=3,
        min_samples=2,
    )

    print("\n3. Dangerous zones:")
    print("Number of polygons:", len(polygons))

    for i, polygon in enumerate(polygons):
        print(f"\nZone {i + 1}")
        print("Area:", polygon.area)
        print("Valid:", polygon.is_valid)
        print("Coordinates:")
        print(list(polygon.exterior.coords))

    print("\n=== TEST COMPLETED ===")


if __name__ == "__main__":
    main()